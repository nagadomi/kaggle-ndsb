if nn.LeakyReLU then
   return
end
local LeakyReLU, parent = torch.class('nn.LeakyReLU','nn.Module')
 
function LeakyReLU:__init(negative_scale)
   parent.__init(self)
   self.negative_scale = negative_scale or 0.333
   self.positive = torch.Tensor()
   self.negative = torch.Tensor()
   self.filter_positive = torch.Tensor()
   self.filter_negative = torch.Tensor()
end
 
function LeakyReLU:updateOutput(input)
   self.output:resizeAs(input)
   self.positive:resizeAs(input)
   self.negative:resizeAs(input)
   
   -- self.output[self.output:lt(0)]:mul(self.negative_scale) does not work on CUDA
   torch.gt(self.filter_positive, input, 0):typeAs(input)
   torch.lt(self.filter_negative, input, 0):typeAs(input)
   torch.cmul(self.positive, input, self.filter_positive)
   torch.cmul(self.negative, input, self.filter_negative)
   self.negative:mul(self.negative_scale)
   self.output:copy(self.positive):add(self.negative)
   
   return self.output
end
 
function LeakyReLU:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(gradOutput)
   
   torch.cmul(self.positive, gradOutput, self.filter_positive)
   torch.cmul(self.negative, gradOutput, self.filter_negative)
   self.negative:mul(self.negative_scale)
   self.gradInput:copy(self.positive):add(self.negative)
   
   return self.gradInput
end
