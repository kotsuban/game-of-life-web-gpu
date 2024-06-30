import './style/style.css'

const GRID_SIZE = 16
const UPDATE_INTERVAL = 1000
const WORKGROUP_SIZE = 8

async function prepareAdapter() {
  if (!navigator.gpu) {
    return console.error("This browser don't support WebGPU.")
  }

  const adapter = await navigator.gpu.requestAdapter()

  if (!adapter) {
    return console.error('No appropriate GPUAdapter found.')
  }

  return adapter
}

let step = 0

async function init() {
  const adapter = await prepareAdapter()

  const device = await adapter.requestDevice()
  const canvasFormat = navigator.gpu.getPreferredCanvasFormat()
  const canvas = document.getElementById('board')
  const context = canvas.getContext('webgpu')

  context.configure({ device: device, format: canvasFormat })

  const vertices = new Float32Array([
    -0.8, -0.8, 0.8, -0.8, 0.8, 0.8,

    -0.8, -0.8, 0.8, 0.8, -0.8, 0.8,
  ])
  const vertexBuffer = device.createBuffer({
    label: 'Cell vertices',
    size: vertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  })

  const uniformArray = new Float32Array([GRID_SIZE, GRID_SIZE])
  const uniformBuffer = device.createBuffer({
    label: 'Grid uniforms',
    size: uniformArray.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })

  const cellStateArray = new Uint32Array(GRID_SIZE * GRID_SIZE)
  const cellStateStorage = [
    device.createBuffer({
      label: 'Cell State A',
      size: cellStateArray.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    }),
    device.createBuffer({
      label: 'Cell State B',
      size: cellStateArray.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    }),
  ]

  device.queue.writeBuffer(vertexBuffer, 0, vertices)
  device.queue.writeBuffer(uniformBuffer, 0, uniformArray)

  for (let i = 0; i < cellStateArray.length; ++i) {
    cellStateArray[i] = Math.random() > 0.8 ? 1 : 0
  }
  device.queue.writeBuffer(cellStateStorage[0], 0, cellStateArray)

  const vertexBufferLayout = {
    arrayStride: 8,
    attributes: [
      {
        format: 'float32x2',
        offset: 0,
        shaderLocation: 0, // Position, see vertex shader
      },
    ],
  }

  const cellShaderModule = device.createShaderModule({
    label: 'Cell shader',
    code: `
      struct VertexInput {
        @location(0) pos: vec2f,
        @builtin(instance_index) instance: u32,
      };

      struct VertexOutput {
        @builtin(position) pos: vec4f,
        @location(0) cell: vec2f,
      };

      @group(0) @binding(0) var<uniform> grid: vec2f;
      @group(0) @binding(1) var<storage> cellState: array<u32>;

      @vertex
      fn vertexMain(input: VertexInput) -> VertexOutput {
        let i = f32(input.instance);
        let cell = vec2f(i % grid.x, floor(i / grid.x));
        let state = f32(cellState[input.instance]);
        let cellOffset = cell / grid * 2;
        let gridPos = (input.pos * state + 1) / grid - 1 + cellOffset;

        var output: VertexOutput;
        output.pos = vec4f(gridPos, 0, 1);
        output.cell = cell;

        return output;
      }

      @fragment
      fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
        // return vec4f(0.79, 0.65, 0.96, 1); // (R, G, B, A)
        let c = input.cell / grid;
        return vec4f(c, 1 - c.x, 1);
      }
    `,
  })

  const bindGroupLayout = device.createBindGroupLayout({
    label: 'Cell Bind Group Layout',
    entries: [
      {
        binding: 0,
        visibility:
          GPUShaderStage.FRAGMENT |
          GPUShaderStage.VERTEX |
          GPUShaderStage.COMPUTE,
        buffer: {},
      },
      {
        binding: 1,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE,
        buffer: { type: 'read-only-storage' },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' },
      },
    ],
  })

  const pipelineLayout = device.createPipelineLayout({
    label: 'Cell Pipeline Layout',
    bindGroupLayouts: [bindGroupLayout],
  })

  const cellPipeline = device.createRenderPipeline({
    label: 'Cell pipeline',
    layout: pipelineLayout,
    vertex: {
      module: cellShaderModule,
      entryPoint: 'vertexMain',
      buffers: [vertexBufferLayout],
    },
    fragment: {
      module: cellShaderModule,
      entryPoint: 'fragmentMain',
      targets: [
        {
          format: canvasFormat,
        },
      ],
    },
  })

  const simulationShaderModule = device.createShaderModule({
    label: 'Game of Life simulation shader',
    code: `
      @group(0) @binding(0) var<uniform> grid: vec2f; 
      @group(0) @binding(1) var<storage> cellStateIn: array<u32>;
      @group(0) @binding(2) var<storage, read_write> cellStateOut: array<u32>;

      fn cellIndex(cell: vec2u) -> u32 {
        return (cell.y % u32(grid.y)) * u32(grid.x) +
               (cell.x % u32(grid.x));
      }

      fn cellActive(x: u32, y: u32) -> u32 {
        return cellStateIn[cellIndex(vec2(x, y))];
      }

      @compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
      fn computeMain(@builtin(global_invocation_id) cell: vec3u) {
        let activeNeighbors = cellActive(cell.x + 1, cell.y + 1) +
                              cellActive(cell.x + 1, cell.y) +
                              cellActive(cell.x + 1, cell.y - 1) +
                              cellActive(cell.x, cell.y - 1) +
                              cellActive(cell.x - 1, cell.y - 1) +
                              cellActive(cell.x - 1, cell.y) +
                              cellActive(cell.x - 1, cell.y + 1) +
                              cellActive(cell.x, cell.y + 1);
        let i = cellIndex(cell.xy);

        switch activeNeighbors {
          case 2: {
            cellStateOut[i] = cellStateIn[i];
          }
          case 3: {
            cellStateOut[i] = 1;
          }
          default: {
            cellStateOut[i] = 0;
          }
        }
      }
    `,
  })

  const simulationPipeline = device.createComputePipeline({
    label: 'Simulation pipeline',
    layout: pipelineLayout,
    compute: {
      module: simulationShaderModule,
      entryPoint: 'computeMain',
    },
  })

  const bindGroups = [
    device.createBindGroup({
      label: 'Cell renderer bind group A',
      layout: bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: { buffer: uniformBuffer },
        },
        {
          binding: 1,
          resource: { buffer: cellStateStorage[0] },
        },
        {
          binding: 2,
          resource: { buffer: cellStateStorage[1] },
        },
      ],
    }),
    device.createBindGroup({
      label: 'Cell renderer bind group B',
      layout: bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: { buffer: uniformBuffer },
        },
        {
          binding: 1,
          resource: { buffer: cellStateStorage[1] },
        },
        {
          binding: 2,
          resource: { buffer: cellStateStorage[0] },
        },
      ],
    }),
  ]

  function update() {
    const encoder = device.createCommandEncoder()
    const computePass = encoder.beginComputePass()

    computePass.setPipeline(simulationPipeline)
    computePass.setBindGroup(0, bindGroups[step % 2])

    const workgroundCount = Math.ceil(GRID_SIZE / WORKGROUP_SIZE)
    computePass.dispatchWorkgroups(workgroundCount, workgroundCount)

    computePass.end()

    step++
    console.log('t', step % 2)
    const renderPass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          loadOp: 'clear',
          clearValue: { r: 0.11, g: 0.11, b: 0.18, a: 1 },
          storeOp: 'store',
        },
      ],
    })

    renderPass.setPipeline(cellPipeline)
    renderPass.setBindGroup(0, bindGroups[step % 2])
    renderPass.setVertexBuffer(0, vertexBuffer)
    renderPass.draw(vertices.length / 2, GRID_SIZE * GRID_SIZE) // 6 vertices

    renderPass.end()
    device.queue.submit([encoder.finish()])
  }

  setInterval(update, UPDATE_INTERVAL)
}

init()
