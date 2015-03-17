require 'cutorch'
require 'cunn'
require './util'

local settings = require './settings'
local preprocess = require './preprocess'
local transform = require './transform'

local function predict(file, model, test_x)
   local step = 68
   local fp = io.open(file, "w")

   for i = 1, #test_x do
      local preds = torch.Tensor(settings.classes):zero()
      local x = transform.augment_accurate(test_x[i], settings.image_size)
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
      
      for k = 1, preds:size(1) do
	 if k ~= 1 then
	    fp:write(",")
	 end
	 fp:write(preds[k])
      end
      fp:write("\n")
      xlua.progress(i, #test_x)
   end
   xlua.progress(#test_x, #test_x)
   fp:close()
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

local function cv_prediction()
   local x = torch.load(string.format("%s/train_x.t7", settings.data_dir))
   local y = torch.load(string.format("%s/train_y.t7", settings.data_dir))
   local train_x, train_y, valid_x, valid_y = split_data(x, y, {3000, 0.1})
   local model = torch.load(settings.model_file)
   local file = string.format("%s/cv_%dx_%s_%d.txt",
			      settings.model_dir,
			      settings.model,
			      settings.split,
			      settings.seed)
   model:evaluate()
   predict(file, model, valid_x)
end
print(settings)
settings.set_seed()
cv_prediction()
