require 'cutorch'
require 'cunn'
require './TrueNLLCriterion'
require './util'

local settings = require './settings'
local minibatch_sgd = require './minibatch_sgd'
local preprocess = require './preprocess'
local iproc = require './iproc'
local transform = require './transform'

local function test(model, valid_x, valid_y, criterion)
   local sum_loss = 0   
   for i = 1, #valid_x do
      local preds = torch.Tensor(settings.classes):zero()
      local x = transform.augment(valid_x[i], settings.image_size)
      local step = 48
      
      x = preprocess.gcn(x)
      for j = 1, x:size(1), step do
	 local batch = torch.Tensor(step, x:size(2), x:size(3), x:size(4)):zero()
	 local n = step
	 if j + n > x:size(1) then
	    n = 1 + n - ((j + n) - x:size(1))
	 end
	 batch:narrow(1, 1, n):copy(x:narrow(1, j, n))
	 local z = model:forward(batch:cuda()):float()
	 -- averaging
	 for k = 1, n do
	    preds = preds + z[k]
	 end
      end
      preds:div(x:size(1))
      sum_loss = sum_loss + criterion:forward(preds, valid_y[i])
      xlua.progress(i, #valid_x)
   end
   xlua.progress(#valid_x, #valid_x)

   return sum_loss / #valid_x
end

local function split_data_unrestricted(x, y, test_size)
   local index = torch.randperm(#x)
   local train_size = #x - test_size
   local train_x = {}
   local train_y = torch.Tensor(train_size):zero()
   local valid_x = {}
   local valid_y = torch.Tensor(test_size):zero()
   for i = 1, train_size do
      train_x[i] = x[index[i]]
      train_y[i] = y[index[i]]
   end
   for i = 1, test_size do
      valid_x[i] = x[index[train_size + i]]
      valid_y[i] = y[index[train_size + i]]
   end
   return train_x, train_y, valid_x, valid_y
end

local function split_data_stratified(x, y, ratio)
   local train_x = {}
   local train_y = {}
   local valid_x = {}
   local valid_y = {}
   local count = {}
   local ti = 1
   local vi = 1

   for i = 1, y:size(1) do
      if count[y[i]] == nil then
	 count[y[i]] = 0
      end
      count[y[i]] = count[y[i]] + 1
   end
   for k, v in pairs(count) do
      local class_list = {}
      for i = 1, y:size(1) do
	 if y[i] == k then
	    table.insert(class_list, i)
	 end
      end
      local test_size = math.max(math.floor(#class_list * ratio), 1)
      local train_size = #class_list - test_size
      local index = torch.randperm(#class_list)
      for i = 1, train_size do
	 train_x[ti] = x[class_list[index[i]]]
	 train_y[ti] = y[class_list[index[i]]]
	 ti = ti + 1
      end
      for i = 1, test_size do
	 valid_x[vi] = x[class_list[index[train_size + i]]]
	 valid_y[vi] = y[class_list[index[train_size + i]]]
	 vi = vi + 1
      end
   end
   train_y = torch.LongTensor(train_y)
   valid_y = torch.LongTensor(valid_y)
   return train_x, train_y, valid_x, valid_y
end
local function split_data(x, y, ratio)
   if settings.split == "unrestricted" then
      return split_data_unrestricted(x, y, ratio[1])
   elseif settings.split == "stratified" then
      return split_data_stratified(x, y, ratio[2])
   else
      assert(0)
   end
end

local function training()
   local MAX_EPOCH = 500
   local FINAL_STAGE = 300
   local VALIDATION_START_EPOCH = 120
   local best_score = 1000.0
   local class2id, id2class = load_classes()
   local x = torch.load(string.format("%s/train_x.t7", settings.data_dir))
   local y = torch.load(string.format("%s/train_y.t7", settings.data_dir))
   local train_x, train_y, valid_x, valid_y = split_data(x, y, {3000, 0.1})
   local model = settings.create_model():cuda()
   local criterion = nn.TrueNLLCriterion():cuda()
   local sgd_config = {
      learningRate = 0.01,
      learningRateDecay = 5.0e-5,
      momentum = 0.9,
      evalCounter = 0,
      xBatchSize = 64,
   }
   local transformer = function (x)
      return preprocess.gcn(transform.random(x, settings.image_size))
   end
   criterion.sizeAverage = false

   for epoch = 1, MAX_EPOCH do
      if epoch == FINAL_STAGE then
	 sgd_config = {
	    learningRate = 0.001,
	    learningRateDecay = 5.0e-5,
	    momentum = 0.9,
	    evalCounter = 0,
	    xBatchSize = 64,
	 }
      end
      model:training()
      print("# " .. epoch)
      print("## train LR: " .. (sgd_config.learningRate / (1 + sgd_config.evalCounter * sgd_config.learningRateDecay)))
      print(minibatch_sgd(model, criterion, train_x, train_y, sgd_config,
			  transformer, {1, settings.image_size, settings.image_size}))
      if epoch >= VALIDATION_START_EPOCH and epoch % 20 == 0 then
	 print("## test")
	 model:evaluate()
	 local score = test(model, valid_x, valid_y, criterion)
	 print("best_score: " .. best_score .. ", current score: " .. score)
	 if score < best_score then
	    best_score = score
	    torch.save(settings.model_file, model)
	 end
      end
      collectgarbage()
   end
   
   -- last epochs
   model = nil
   collectgarbage()
   model = torch.load(settings.model_file)
   for i = 1, 5 do
      sgd_config.learningRate = sgd_config.learningRate * 0.5
      model:training()
      print(minibatch_sgd(model, criterion, train_x, train_y, sgd_config,
			  transformer, {1, settings.image_size, settings.image_size}))
      model:evaluate()
      local score = test(model, valid_x, valid_y, criterion)
      print("best_score: " .. best_score .. ", current score: " .. score)
      if score < best_score then
	 best_score = score
	 torch.save(settings.model_file, model)
      end
   end
end

settings.set_seed()
print("settings =")
print(settings)
training()
