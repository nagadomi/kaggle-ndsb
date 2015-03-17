local create_model = require './cnn_96x96'
require './TrueNLLCriterion'

local function cuda_test()
   local model = create_model():cuda()
   local criterion = nn.TrueNLLCriterion():cuda()
   local x = torch.Tensor(64, 1, 96, 96):uniform():cuda()
   local y = torch.Tensor(64):random(1, 121):cuda()
   local z = model:forward(x)
   local df_do = torch.Tensor(z:size(1), z:size(2)):zero()
   for i = 1, z:size(1) do
      local err = criterion:forward(z[i], y[i])
      df_do[i]:copy(criterion:backward(z[i], y[i]))
   end
   model:backward(x, df_do:cuda())
   print("CUDA Test Successful!")
end

torch.setdefaulttensortype('torch.FloatTensor')
print(cutorch.getDeviceProperties(cutorch.getDevice()))
cuda_test()
