require 'optim'
require 'cutorch'
require 'xlua'

local function minibatch_sgd(model, criterion,
			     train_x, train_y,
			     config, transformer, size)
   local parameters, gradParameters = model:getParameters()
   config = config or {}
   local sum_loss = 0
   local count_loss = 0
   local acc = 0
   local batch_size = config.xBatchSize or 32
   local shuffle = torch.randperm(#train_x)
   local c = 1
   
   for t = 1, #train_x, batch_size do
      if t + batch_size > #train_x then
	 break
      end
      xlua.progress(t, #train_x)
      local inputs = torch.Tensor(batch_size,
				  size[1],
				  size[2],
				  size[3])
      local targets = torch.Tensor(batch_size)
      for i = 1, batch_size do
         inputs[i]:copy(transformer(train_x[shuffle[t + i - 1]]))
	 targets[i] = train_y[shuffle[t + i - 1]]
      end
      inputs = inputs:cuda()
      targets = targets:cuda()
      local feval = function(x)
	 if x ~= parameters then
	    parameters:copy(x)
	 end
	 gradParameters:zero()
	 local f = 0
	 local output = model:forward(inputs)
	 local df_do = torch.CudaTensor(output:size(1), output:size(2))
	 for k = 1, output:size(1) do
	    local err = criterion:forward(output[k], targets[k])
	    f = f + err
	    df_do[k]:copy(criterion:backward(output[k], targets[k]))
	    count_loss = count_loss + 1
	    sum_loss = sum_loss + err
	    
	    local max_i1 = targets[k]
	    local v2, max_i2 = output[k]:float():max(1)
	    if max_i1 == max_i2[1] then
	       acc = acc + 1.0
	    end
	 end
	 model:backward(inputs, df_do)
	 gradParameters:div(inputs:size(1))
	 f = f / inputs:size(1)
	 return f, gradParameters
      end
      optim.nag(feval, parameters, config)
      
      c = c + 1
      if c % 100 == 0 then
	 collectgarbage()
      end
   end
   xlua.progress(#train_x, #train_x)
   
   return { logloss = sum_loss / count_loss, accuracy = acc / count_loss}
end

return minibatch_sgd
