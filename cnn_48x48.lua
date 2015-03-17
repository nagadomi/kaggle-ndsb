require 'cunn'
require 'cudnn'
require './LeakyReLU'

local ALPHA = 0.333
function cudnn.SpatialConvolution:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = math.sqrt(2 / ((1.0 + ALPHA * ALPHA) * self.kW * self.kH * self.nOutputPlane))
   end
   self.weight:normal(0, stdv)
   self.bias:normal(0, stdv)
end
function nn.Linear:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = math.sqrt(2 / ((1.0 + ALPHA * ALPHA) * self.weight:size(2)))
   end
   self.weight:normal(0, stdv)
   self.bias:normal(0, stdv)
end
local function create_model()
   local model = nn.Sequential() 

   -- input: 1x48x48
   model:add(cudnn.SpatialConvolution(1, 64, 7, 7, 2, 2, 3, 3))
   model:add(nn.LeakyReLU(ALPHA))
   model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))
   
   model:add(cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1))
   model:add(nn.LeakyReLU(ALPHA))
   model:add(cudnn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1))
   model:add(nn.LeakyReLU(ALPHA))
   model:add(cudnn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1))
   model:add(nn.LeakyReLU(ALPHA))
   model:add(cudnn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1))
   model:add(nn.LeakyReLU(ALPHA))
   model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))
   
   model:add(cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1))
   model:add(nn.LeakyReLU(ALPHA))
   model:add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
   model:add(nn.LeakyReLU(ALPHA))
   model:add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
   model:add(nn.LeakyReLU(ALPHA))
   model:add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
   model:add(nn.LeakyReLU(ALPHA))
   model:add(cudnn.SpatialAveragePooling(6, 6, 6, 6))
   
   model:add(nn.View(256))
   model:add(nn.Linear(256, 512))
   model:add(nn.LeakyReLU(ALPHA))
   model:add(nn.Dropout(0.5))
   model:add(nn.Linear(512, 121))
   model:add(nn.SoftMax())
   
   return model
end
--a = create_model()
--model:cuda()
--print(model:forward(torch.Tensor(32, 1, 48, 48):uniform():cuda()):size())

return create_model
