local cmd = torch.CmdLine()

-- Basic options
cmd:option('-style_image', 'examples/inputs/seated-nude.jpg',
           'Style target image')
cmd:option('-style_blend_weights', 'nil')
cmd:option('-content_image', 'examples/inputs/tubingen.jpg',
           'Content target image')
cmd:option('-init_image', 'examples/inputs/brad_pitt.jpg',
           'Initial image')
cmd:option('-image_size', 512, 'Maximum height / width of generated image')
cmd:option('-gpu', 0, 'Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1')

-- Optimization options
cmd:option('-content_weight', 5e0)
cmd:option('-style_weight', 1e2)
cmd:option('-tv_weight', 1e-3)
cmd:option('-num_iterations', 1000)
cmd:option('-normalize_gradients', false)
cmd:option('-init', 'random', 'random|image')
cmd:option('-optimizer', 'lbfgs', 'lbfgs|adam')
cmd:option('-learning_rate', 1e1)

-- Output options
cmd:option('-print_iter', 50)
cmd:option('-save_iter', 100)
cmd:option('-output_image', 'out.png')

-- Other options
cmd:option('-style_scale', 1.0)
cmd:option('-pooling', 'max', 'max|avg')
cmd:option('-proto_file', 'models/VGG_ILSVRC_19_layers_deploy.prototxt')
cmd:option('-model_file', 'models/VGG_ILSVRC_19_layers.caffemodel')
cmd:option('-backend', 'nn', 'nn|cudnn')
cmd:option('-seed', -1)

cmd:option('-content_layers', 'relu4_2', 'layers for content')
cmd:option('-style_layers', 'relu1_1,relu2_1,relu3_1,relu4_1,relu5_1', 'layers for style')

local params = cmd:parse(arg)

local loadcaffe_wrap = require 'loadcaffe_wrapper'
require 'neural_style'

local content_losses, style_losses = {}, {}
local function load_cnn()
  if params.gpu >= 0 then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(params.gpu + 1)
  else
    params.backend = 'nn-cpu'
  end

  if params.backend == 'cudnn' then
    require 'cudnn'
    cudnn.SpatialConvolution.accGradParameters = nn.SpatialConvolutionMM.accGradParameters -- ie: nop
  end
  
  local cnn = loadcaffe_wrap.load(params.proto_file, params.model_file, params.backend):float()
  if params.gpu >= 0 then
    cnn:cuda()
  end
  
  local content_image = image.load(params.content_image, 3)
  content_image = image.scale(content_image, params.image_size, 'bilinear')
  local content_image_caffe = preprocess(content_image):float()
  
  local style_size = math.ceil(params.style_scale * params.image_size)
  local style_image_list = params.style_image:split(',')
  local style_images_caffe = {}
  for _, img_path in ipairs(style_image_list) do
    local img = image.load(img_path, 3)
    img = image.scale(img, style_size, 'bilinear')
    local img_caffe = preprocess(img):float()
    table.insert(style_images_caffe, img_caffe)
  end

  -- Handle style blending weights for multiple style inputs
  local style_blend_weights = nil
  if params.style_blend_weights == 'nil' then
    -- Style blending not specified, so use equal weighting
    style_blend_weights = {}
    for i = 1, #style_image_list do
      table.insert(style_blend_weights, 1.0)
    end
  else
    style_blend_weights = params.style_blend_weights:split(',')
    assert(#style_blend_weights == #style_image_list,
      '-style_blend_weights and -style_images must have the same number of elements')
  end
  -- Normalize the style blending weights so they sum to 1
  local style_blend_sum = 0
  for i = 1, #style_blend_weights do
    style_blend_weights[i] = tonumber(style_blend_weights[i])
    style_blend_sum = style_blend_sum + style_blend_weights[i]
  end
  for i = 1, #style_blend_weights do
    style_blend_weights[i] = style_blend_weights[i] / style_blend_sum
  end
  

  if params.gpu >= 0 then
    content_image_caffe = content_image_caffe:cuda()
    for i = 1, #style_images_caffe do
      style_images_caffe[i] = style_images_caffe[i]:cuda()
    end
  end
  
  local content_layers = params.content_layers:split(",")
  local style_layers = params.style_layers:split(",")

  -- Set up the network, inserting style and content loss modules
  -- local content_losses, style_losses = {}, {}
  local next_content_idx, next_style_idx = 1, 1
  local net = nn.Sequential()
  if params.tv_weight > 0 then
    local tv_mod = nn.TVLoss(params.tv_weight):float()
    if params.gpu >= 0 then
      tv_mod:cuda()
    end
    net:add(tv_mod)
  end
  for i = 1, #cnn do
    if next_content_idx <= #content_layers or next_style_idx <= #style_layers then
      local layer = cnn:get(i)
      local name = layer.name
      local layer_type = torch.type(layer)
      local is_pooling = (layer_type == 'cudnn.SpatialMaxPooling' or layer_type == 'nn.SpatialMaxPooling')
      if is_pooling and params.pooling == 'avg' then
        assert(layer.padW == 0 and layer.padH == 0)
        local kW, kH = layer.kW, layer.kH
        local dW, dH = layer.dW, layer.dH
        local avg_pool_layer = nn.SpatialAveragePooling(kW, kH, dW, dH):float()
        if params.gpu >= 0 then avg_pool_layer:cuda() end
        local msg = 'Replacing max pooling at layer %d with average pooling'
        print(string.format(msg, i))
        net:add(avg_pool_layer)
      else
        net:add(layer)
      end
      if name == content_layers[next_content_idx] then
        print("Setting up content layer", i, ":", layer.name)
        local target = net:forward(content_image_caffe):clone()
        local norm = params.normalize_gradients
        local loss_module = nn.ContentLoss(params.content_weight, target, norm):float()
        if params.gpu >= 0 then
          loss_module:cuda()
        end
        net:add(loss_module)
        table.insert(content_losses, loss_module)
        next_content_idx = next_content_idx + 1
      end
      if name == style_layers[next_style_idx] then
        print("Setting up style layer  ", i, ":", layer.name)
        local gram = GramMatrix():float()
        if params.gpu >= 0 then
          gram = gram:cuda()
        end
        local target = nil
        for i = 1, #style_images_caffe do
          local target_features = net:forward(style_images_caffe[i]):clone()
          local target_i = gram:forward(target_features):clone()
          target_i:div(target_features:nElement())
          target_i:mul(style_blend_weights[i])
          if i == 1 then
            target = target_i
          else
            target:add(target_i)
          end
        end
        local norm = params.normalize_gradients
        local loss_module = nn.StyleLoss(params.style_weight, target, norm):float()
        if params.gpu >= 0 then
          loss_module:cuda()
        end
        net:add(loss_module)
        table.insert(style_losses, loss_module)
        next_style_idx = next_style_idx + 1
      end
    end
  end

  -- We don't need the base CNN anymore, so clean it up to save memory.
  cnn = nil
  for i=1,#net.modules do
    local module = net.modules[i]
    if torch.type(module) == 'nn.SpatialConvolutionMM' then
        -- remote these, not used, but uses gpu memory
        module.gradWeight = nil
        module.gradBias = nil
    end
  end
  collectgarbage()

  return net
--[[  
  -- Initialize the image
  if params.seed >= 0 then
    torch.manualSeed(params.seed)
  end
  local img = nil
  if params.init == 'random' then
    img = torch.randn(content_image:size()):float():mul(0.001)
  elseif params.init == 'image' then
    img = content_image_caffe:clone():float()
  else
    error('Invalid init type')
  end
  if params.gpu >= 0 then
    img = img:cuda()
  end
  
  -- Run it through the network once to get the proper size for the gradient
  -- All the gradients will come from the extra loss modules, so we just pass
  -- zeros into the top of the net on the backward pass.
  local y = net:forward(img)
  local dy = img.new(#y):zero()
--]]
end

local function maybe_print(t, loss)
  local verbose = (params.print_iter > 0 and t % params.print_iter == 0)
  --if verbose then
  if true then
    print(string.format('Iteration %d / %d', t, params.num_iterations))
    for i, loss_module in ipairs(content_losses) do
      print(string.format('  Content %d loss: %f', i, loss_module.loss))
    end
    for i, loss_module in ipairs(style_losses) do
      print(string.format('  Style %d loss: %f', i, loss_module.loss))
    end
    print(string.format('  Total loss: %f', loss))
  end
end

local function maybe_save(t)
  local should_save = params.save_iter > 0 and t % params.save_iter == 0
  should_save = should_save or t == params.num_iterations
  if should_save then
    local disp = deprocess(img:double())
    disp = image.minmax{tensor=disp, min=0, max=1}
    local filename = build_filename(params.output_image, t)
    if t == params.num_iterations then
      filename = params.output_image
    end
    image.save(filename, disp)
  end
end

local num_calls = 0
local function feval(x, net)
  num_calls = num_calls + 1
  net:forward(x)
  local loss = 0
  for _, mod in ipairs(content_losses) do
    loss = loss + mod.loss
  end
  for _, mod in ipairs(style_losses) do
    loss = loss + mod.loss
  end
  --maybe_print(num_calls, loss)
  --maybe_save(num_calls)

  collectgarbage()
  return loss
end

local function init_population(n_ell, n_init)
  local population = {}
  for i = 1, n_init do
    local chromo = torch.randn(n_ell):float():mul(0.001)
    if params.gpu >= 0 then
      chromo = chromo:cuda()
    end
    table.insert(population, chromo)
  end

  return population
end

local function selection(pop, net)
  local new_pop, fitnesses = {}, {}
  local total_f = 0
  local min_f
  for _, chromo in ipairs(pop) do
    table.insert(fitnesses, feval(chromo, net))
  end

  local s = 2 -- selection pressure
  for p = 1, #pop do
    local candidates = torch.randperm(#pop)
    local winner = candidates[1]
    for i = 2, s do
      local challenger = candidates[i]
      if fitnesses[winner] > fitnesses[challenger] then
        winner = challenger
      end
    end
    table.insert(new_pop, winner)

    if min_f == nil or min_f > fitnesses[winner] then min_f = fitnesses[winner] end
    total_f = total_f + fitnesses[winner]
  end

  print("New mean fitness: "..(total_f / #pop))
  print("New min fitness: "..min_f)
  return new_pop
end

local function get_mean_f(pop, net)
  local total_f = 0
  local min_f
  for _, chromo in ipairs(pop) do
    local f = feval(chromo, net)
    total_f = total_f + f
    if min_f == nil or min_f > f then min_f = f end
  end

  print(min_f)
  return total_f / #pop
end

local function extended_line_XO(P1,P2)
  a = torch.rand(2):float():mul(1.25)
  C1 = P1:mul(a[1]):add(P2:mul(1-a[1]))
  C2 = P1:mul(a[2]):add(P2:mul(1-a[2]))
  return {C1,C2}
end

local function BLX_alpha(P1,P2)
  a = 0.5 -- BLX-0.5
  s = P1:size()
  -- P = torch.Tensor(2,s[1],s[2])
  if s:size() == 2 then
    P = torch.repeatTensor(P1,2,1,1)
  else
    P = torch.repeatTensor(P1,2,1)
  end
  -- print (s:size())
  -- P = torch.cat(P1,P2,1)
  P[2] = P2
  max = torch.max(P,1)
  min = torch.min(P,1)
  I = torch.Tensor(s)
  I = torch.add(max, -1, min)
  -- print (I)
  max:add(0.5,I)
  min:add(-0.5,I)
  -- print (max)
  -- print (min)
  -- idx = torch.rand(s):mul(2):int()
  mask = torch.ByteTensor(s):bernoulli()
  -- print (mask)
  
  C1 = max:clone()
  C1:maskedCopy(mask,min)
  C2 = min:clone()
  C2:maskedCopy(mask,max)
  -- print (torch.ne(mask,1))
  -- print (C1)
  -- print (C2)
  return {C1,C2}
end

local function crossover(P1,P2)
  -- C =  extended_line_XO(P1,P2)
  C =  BLX_alpha(P1,P2)
end

local function main()
  local net = load_cnn()

  local content_image = image.load(params.content_image, 3)
  content_image = image.scale(content_image, params.image_size, 'bilinear')
  local image_file = image.load(params.init_image, 3)
  image_file = image.scale(image_file, params.image_size, 'bilinear')
  local image_caffe = preprocess(image_file):float()

  -- Initialize the image
  if params.seed >= 0 then
    torch.manualSeed(params.seed)
  end

  local n_ell = content_image:size() -- 3 * w * l
  local n_init = 1000 -- LOL we are doomed QQ
  local n_current
  local n_pc = 1 -- always crossover ... why not?
  local n_pm = 0
  local n_max_gen = 10
  local sele_pressure = 2

  print("Init population...")
  local population = init_population(n_ell, n_init)
  print("Get mean fitness...")
  print(get_mean_f(population, net))
  print("Selection...")
  local new_pop = selection(population, net)

--[[
  local img = nil
  if params.init == 'random' then
    img = torch.randn(content_image:size()):float():mul(0.001)
  elseif params.init == 'image' then
    img = image_caffe:clone():float()
  else
    error('Invalid init type')
  end
  if params.gpu >= 0 then
    img = img:cuda()
  end

  print('==================')
  print(feval(img, net))

--]]
--
end

local function test()
  X1 = torch.Tensor(1, 14):fill(2)
  X2 = torch.Tensor(1, 14):fill(3)
  crossover(X1,X2)
end


-- main()
test()
