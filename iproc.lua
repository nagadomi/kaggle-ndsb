require 'image'

-- image processing functions

local iproc = {}

local function rotate_with_warp(src, dst, theta, mode)
  local height
  local width
  if src:dim() == 2 then
    height = src:size(1)
    width = src:size(2)
  elseif src:dim() == 3 then
    height = src:size(2)
    width = src:size(3)
  else
    dok.error('src image must be 2D or 3D', 'image.rotate')
  end
  local flow = torch.Tensor(2, height, width)
  local kernel = torch.Tensor({{math.cos(-theta), -math.sin(-theta)},
			       {math.sin(-theta), math.cos(-theta)}})
  flow[1] = torch.ger(torch.linspace(0, 1, height), torch.ones(width))
  flow[1]:mul(-(height -1)):add(math.floor(height / 2 + 0.5))
  flow[2] = torch.ger(torch.ones(height), torch.linspace(0, 1, width))
  flow[2]:mul(-(width -1)):add(math.floor(width / 2 + 0.5))
  flow:add(-1, torch.mm(kernel, flow:view(2, height * width)))
  
  dst:resizeAs(src)
  return image.warp(dst, src, flow, mode, true, 'pad')
end

local function scale_with_warp(src, dst, mode)
   local src_height, src_width, dst_height, dst_width
   if src:dim() == 2 then
      src_height = src:size(1)
      src_width = src:size(2)
   elseif src:dim() == 3 then
      src_height = src:size(2)
      src_width = src:size(3)
   else
      dok.error('src image must be 2D or 3D', 'image.rotate')
   end
   if dst:dim() == 2 then
      dst_height = dst:size(1)
      dst_width = dst:size(2)
   elseif dst:dim() == 3 then
      dst_height = dst:size(2)
      dst_width = dst:size(3)
   else
      dok.error('src image must be 2D or 3D', 'image.rotate')
   end
   local flow = torch.Tensor(2, dst_height, dst_width)
   flow[1] = torch.ger(torch.linspace(0, src_height - 1, dst_height), torch.ones(dst_width))
   flow[2] = torch.ger(torch.ones(dst_height), torch.linspace(0, src_width - 1, dst_width))
   return image.warp(dst, src, flow, mode, false)
end

function iproc.nega(img)
   return -img + 1
end

function iproc.scale(src, width, height)
   local dst = torch.Tensor(src:size(1), height, width)
   scale_with_warp(src, dst, 'bicubic')
   dst[torch.lt(dst, 0)] = 0
   return dst
end

function iproc.rotate(src, theta)
   local dst = torch.Tensor():resizeAs(src)
   rotate_with_warp(src, dst, theta, 'bicubic')
   dst[torch.lt(dst, 0)] = 0
   return dst
end

function iproc.scale_contrast(img, scale)
   img = img:clone()
   img:mul(scale)
   img[torch.gt(img, 1.0)] = 1.0
   return img
end

function iproc.zero_padding(img, x, y)
   y = y or x
   local new_image = torch.Tensor(1, img:size(2) + x * 2, img:size(3) + y * 2):zero()
   new_image[{{}, {x + 1, -x - 1}, {y + 1, -y - 1}}]:copy(img)
   return new_image
end

function iproc.gcn(img)
   local mean = img:mean()
   local std = img:std() + 1.0e-7
   return (img - mean):div(std)
end

local function get_perspective_param(p)
   local params = torch.Tensor(8)
   local sx = p[1].x - p[2].x + p[3].x - p[4].x
   local sy = p[1].y - p[2].y + p[3].y - p[4].y
   local dx1 = p[2].x - p[3].x
   local dy1 = p[2].y - p[3].y
   local dx2 = p[4].x - p[3].x
   local dy2 = p[4].y - p[3].y
   local z = dx1 * dy2 - dy1 * dx2
   local g = (sx * dy2 - sy * dx2) / z
   local h = (sy * dx1 - sx * dy1) / z

   params[1] = p[2].x - p[1].x + g * p[2].x
   params[2] = p[4].x - p[1].x + h * p[4].x
   params[3] = p[1].x
   params[4] = p[2].y - p[1].y + g * p[2].y
   params[5] = p[4].y - p[1].y + h * p[4].y
   params[6] = p[1].y
   params[7] = g
   params[8] = h
   
   return params
end
function iproc.perspective_crop(src, x1, y1, x2, y2, x3, y3, x4, y4, w, h)
   if src:dim() ~= 3 then
      error("expected 3d tensor")
   end
   local height = h or src:size(2)
   local width = w or src:size(3)
   local flow = torch.Tensor(2, height, width)
   local p = get_perspective_param({{x = x1, y = y1}, -- top left
				    {x = x2, y = y2}, -- top right
				    {x = x3, y = y3}, -- bottom right
				    {x = x4, y = y4}  -- bottom left
				   })
   local v = torch.ger(torch.linspace(-1,1, height), torch.ones(width)):add(1):mul(0.5)
   local u = torch.ger(torch.ones(height), torch.linspace(-1,1, width)):add(1):mul(0.5)
   local t = (u * p[7]) + (v * p[8]) + 1
   flow[1]:copy(u * p[4]):add(v * p[5]):add(p[6]):cdiv(t)
   flow[2]:copy(u * p[1]):add(v * p[2]):add(p[3]):cdiv(t)
   dst = torch.Tensor(1, h, w)
   image.warp(dst, src, flow, 'bicubic', false, 'pad')
   dst[torch.lt(dst, 0)] = 0
   return dst
end

local function test_perspective_crop()
   torch.manualSeed(10)
   local src = image.lena():narrow(1, 1, 1)
   local imgs = torch.Tensor(25, 1, 512, 512)
   for i = 1, 25 do
      imgs[i]:copy(
	 iproc.perspective_crop(src,
				1 + torch.random(0, 40), 1 + torch.random(0, 40),
				512 - torch.random(0, 40), 1 + torch.random(0, 40),
				512 - torch.random(0, 40), 512 - torch.random(0, 40),
				1 + torch.random(0, 40), 512 - torch.random(0, 40))
		  )
   end
   imgs[10] = src
   local view = image.toDisplayTensor({input = imgs,
				       padding = 2,
				       nrow = 5})
   image.save("lena.png", image.scale(view, 800, 800))
end

--test_perspective_crop()

return iproc
