local preprocess = require './preprocess'
local settings = require './settings'

require './util'

local function count_lines(file)
   local fp = io.open(file, "r")
   local count = 0
   for line in fp:lines() do
      count = count + 1
   end
   fp:close()
   
   return count
end

local function filename_to_label(filename)
   local names = utils.split(filename, "/")
   return names[#names-1]
end

local function convert_train()
   local class2id, id2label = load_classes()
   local list_file = string.format("%s/train.txt", settings.data_dir)
   local train_n = count_lines(list_file)
   local x = {}
   local y = torch.LongTensor(train_n)
   local fp = io.open(list_file, "r")
   local line
   local i = 1
   for filename in fp:lines() do
      local label = filename_to_label(filename)
      local img = image.load(filename)
      x[i] = preprocess.reconstruct(img)
      y[i] = class2id[label]
      if i % 100 == 0 then
	 xlua.progress(i, train_n)
	 collectgarbage()
      end
      i = i + 1
   end
   xlua.progress(train_n, train_n)
   fp:close()
   
   torch.save(string.format("%s/train_x.t7", settings.data_dir), x)
   torch.save(string.format("%s/train_y.t7", settings.data_dir), y)
end

local function convert_test()
   local x = {}
   local list_file = string.format("%s/test.txt", settings.data_dir)
   local test_n = count_lines(list_file)
   local file = io.open(list_file, "r")
   local line
   local i = 1
   for filename in file:lines() do
      local img = image.load(filename)
      x[i] = preprocess.reconstruct(img)
      if i % 100 == 0 then
	 xlua.progress(i, test_n)
	 collectgarbage()
      end
      i = i + 1
   end
   xlua.progress(test_n, test_n)
   file:close()
   torch.save(string.format("%s/test_x.t7", settings.data_dir), x)
end

print("convert train data ...")
convert_train()
collectgarbage()
print("convert test data ...")
convert_test()
