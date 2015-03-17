require 'cutorch'
require 'cunn'
require './TrueNLLCriterion'
require './util'

local settings = require './settings'
local CV = 8

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

local function load_pred(file)
   local fp = io.open(file, "r")
   local preds = {}
   for line in fp:lines() do
      local cols = utils.split(line, ",")
      local vec = {}
      for i = 1, #cols do
	 vec[i] = tonumber(cols[i])
      end
      table.insert(preds, vec)
   end
   fp:close()
   
   return torch.Tensor(preds)
end

local function load_data(model, x, y)
   local preds = {}
   local ys = {}
   for i = 1, CV do
      settings.seed = 100 + i
      settings.set_seed()
      local train_x, train_y, valid_x, valid_y = split_data_stratified(x, y, 0.1)
      local file = string.format("%s/cv_%dx_%s_%d.txt",
				 settings.model_dir,
				 model,
				 settings.split,
				 settings.seed)
      table.insert(preds, load_pred(file))
      table.insert(ys, valid_y)
      collectgarbage()
   end
   return preds, ys
end

local function validate(pred48, pred72, pred96, y, w, calb)
   local criterion = nn.TrueNLLCriterion()
   criterion.sizeAverage = false
   local sum_logloss = 0
   
   for i = 1, y:size(1) do
      pred48[i]:div(pred48[i]:sum())
      pred72[i]:div(pred72[i]:sum())
      pred96[i]:div(pred96[i]:sum())
      local z = (pred48[i] * w[1] + pred72[i] * w[2] + pred96[i] * w[3])
      z:add(calb)
      z:div(z:sum())
      local loss = criterion:forward(z, y[i])
      sum_logloss = sum_logloss + loss
   end
   return sum_logloss / y:size(1)
end

local function run()
   local x = torch.load(string.format("%s/train_x.t7", settings.data_dir))
   local y = torch.load(string.format("%s/train_y.t7", settings.data_dir))
   local pred48, ys = load_data(48, x, y)
   local pred72, _ = load_data(72, x, y)   
   local pred96, _ = load_data(96, x, y)
   local best_loss = 10000
   local rand_base = {1.0, 1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-90}
   
   -- brute force attack with random parameter
   for i = 1, 1000 do
      local ls = {}
      local loss = 0.0
      local w = torch.Tensor(3):uniform()
      local base = rand_base[torch.random(1, #rand_base)]
      local calb = torch.uniform(0, base)
      w:div(w:sum())
      for k = 1, CV do
	 ls[k] = validate(pred48[k], pred72[k], pred96[k], ys[k], w, calb)
	 loss = loss + ls[k]
      end
      loss = loss / CV
      if best_loss > loss then
	 print(loss, w[1], w[2], w[3], calb)
	 best_loss = loss
      end
      collectgarbage()
   end
end

run()
--  with   calib: 0.61635209461085 0.22928008437157 0.34947738051414 0.42124256491661 9.8288448294625e-06
-- without calib: 0.61796191851043 0.22799848020077 0.34784850478172 0.42415297031403
