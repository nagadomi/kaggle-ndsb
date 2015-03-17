require 'cunn'

-- ref: https://github.com/torch/nn/issues/112#issuecomment-64427049
local function zeroDataSize(data)
   if type(data) == 'table' then
      for i = 1, #data do
	 data[i] = zeroDataSize(data[i])
      end
   elseif type(data) == 'userdata' then
      data = torch.Tensor():typeAs(data)
   end
   return data
end

-- Resize the output, gradInput, etc temporary tensors to zero (so that the
-- on disk size is smaller)
local function cleanupModel(node)
   if node.output ~= nil then
      node.output = zeroDataSize(node.output)
   end
   if node.gradInput ~= nil then
      node.gradInput = zeroDataSize(node.gradInput)
   end
   if node.finput ~= nil then
      node.finput = zeroDataSize(node.finput)
   end
   if tostring(node) == "nn.LeakyReLU" then
      if node.positive ~= nil then
	 node.positive = zeroDataSize(node.positive)
      end
      if node.negative ~= nil then
	 node.negative = zeroDataSize(node.negative)
      end
      if node.filter_positive ~= nil then
	 node.filter_positive = zeroDataSize(node.filter_positive)
      end
      if node.filter_negative ~= nil then
	 node.filter_negative = zeroDataSize(node.filter_negative)
      end
   end
   if tostring(node) == "nn.Dropout" then
      if node.noise ~= nil then
	 node.noise = zeroDataSize(node.noise)
      end
   end
   -- Recurse on nodes with 'modules'
   if (node.modules ~= nil) then
     if (type(node.modules) == 'table') then
	for i = 1, #node.modules do
	   local child = node.modules[i]
	   cleanupModel(child)
	end
     end
   end
   
   collectgarbage()
end

local cmd = torch.CmdLine()
cmd:text()
cmd:text("cleanup model")
cmd:text("Options:")
cmd:option("-path", "./model.t7", 'path of model')
local opt = cmd:parse(arg)
local model = torch.load(opt.path)
if model then
   cleanupModel(model)
   torch.save(opt.path, model)
else
   error("model not found")
end
