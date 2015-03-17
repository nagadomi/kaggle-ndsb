require 'cunn'
require 'cudnn'
require './LeakyReLU'

-- this architecture inspired by the work of http://arxiv.org/abs/1502.01852

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

   -- input: 1x96x96
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
   model:add(nn.Dropout(0.25))

   model:add(cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1))
   model:add(nn.LeakyReLU(ALPHA))
   model:add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
   model:add(nn.LeakyReLU(ALPHA))
   model:add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
   model:add(nn.LeakyReLU(ALPHA))
   model:add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
   model:add(nn.LeakyReLU(ALPHA))
   model:add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
   model:add(nn.LeakyReLU(ALPHA))
   model:add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
   model:add(nn.LeakyReLU(ALPHA))
   model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))
   model:add(nn.Dropout(0.25))
   
   model:add(cudnn.SpatialConvolution(256, 320, 3, 3, 1, 1, 1, 1))
   model:add(nn.LeakyReLU(ALPHA))
   model:add(cudnn.SpatialConvolution(320, 320, 3, 3, 1, 1, 1, 1))
   model:add(nn.LeakyReLU(ALPHA))
   model:add(cudnn.SpatialConvolution(320, 320, 3, 3, 1, 1, 1, 1))
   model:add(nn.LeakyReLU(ALPHA))
   model:add(cudnn.SpatialConvolution(320, 320, 3, 3, 1, 1, 1, 1))
   model:add(nn.LeakyReLU(ALPHA))
   model:add(cudnn.SpatialConvolution(320, 320, 3, 3, 1, 1, 1, 1))
   model:add(nn.LeakyReLU(ALPHA))
   model:add(cudnn.SpatialConvolution(320, 320, 3, 3, 1, 1, 1, 1))
   model:add(nn.LeakyReLU(ALPHA))
   model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))
   model:add(nn.Dropout(0.25))
   
   model:add(nn.View(320 * 3 * 3))
   model:add(nn.Linear(320 * 3 * 3, 1024))
   model:add(nn.LeakyReLU(ALPHA))
   model:add(nn.Dropout(0.5))
   model:add(nn.Linear(1024, 1024))
   model:add(nn.LeakyReLU(ALPHA))
   model:add(nn.Dropout(0.5))
   model:add(nn.Linear(1024, 121))
   model:add(nn.SoftMax())
   
   return model
end

return create_model
