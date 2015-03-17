require 'cutorch'
require 'cunn'
require './util'

local settings = require './settings'
local preprocess = require './preprocess'
local transform = require './transform'

local function predict(file, model, test_x, filenames)
   local class2id, id2class = load_classes()   
   local step = 68
   local fp = io.open(file, "w")

   fp:write("image")
   for i = 1, #id2class do
      fp:write(",")
      fp:write(id2class[i])
   end
   fp:write("\n")
   
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
      
      fp:write(filenames[i] .. ",")
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
local function load_filenames()
   local fp = io.open(string.format("%s/test.txt", settings.data_dir))
   local filenames = {}
   local i = 1
   for line in fp:lines() do
      local names = utils.split(line, "/")
      filenames[i] = names[#names]
      i = i + 1
   end
   return filenames
end

local function prediction()
   local x = torch.load(string.format("%s/test_x.t7", settings.data_dir))
   local model = torch.load(settings.model_file)

   model:evaluate()
   
   predict(settings.submission_file, model, x, load_filenames())
end

print("settings =")
print(settings)
prediction()
