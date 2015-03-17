require 'image'
require 'nn'

-- preprocessing functions

local iproc = require 'iproc'
local preprocess = {}

local function center_pos(img)
   local min_x = nil
   local min_y = nil
   local max_x = 0
   local max_y = 0
   local x_sum = img:view(img:size(2), img:size(3)):sum(1)
   local y_sum = img:view(img:size(2), img:size(3)):sum(2)
   local count = 0
   
   for y = 1, img:size(2) do
      if y_sum[y][1] > 0.0 then
	 if min_y == nil then
	    min_y = y
	 end
	 if max_y < y then
	    max_y = y
	 end
      end
   end
   for x = 1, img:size(3) do
      if x_sum[1][x] > 0.0 then
	 if min_x == nil then
	    min_x = x
	 end
	 if max_x < x then
	    max_x = x
	 end
      end
   end
   return min_x + (max_x - min_x) / 2, min_y + (max_y - min_y) / 2, math.max(max_y - min_y, max_x - min_x)
end
local function centering(img)
   if img:size(2) < img:size(3) then
      img = torch.Tensor():resizeAs(img:transpose(2, 3)):copy(img:transpose(2, 3))
   end
   local padding = 2
   local new_image = torch.Tensor(1,
				  (img:size(2) + padding) * 2,
				  (img:size(2) + padding) * 2):fill(1.0) -- fill white
   local offset = img:size(2) / 2 + padding
   
   new_image[{{},
	      {1 + offset,
	       offset + img:size(2)},
	      {1 + offset + (img:size(2) - img:size(3)) / 2,
	       offset + (img:size(2) - img:size(3)) / 2 + img:size(3)}
	     }]:copy(img)
   new_image = iproc.nega(new_image)
   
   local center_x, center_y, len = center_pos(new_image)
   offset = math.ceil((len / 2 + 1) + padding)
   new_image = image.crop(new_image,
			  center_x - offset,
			  center_y - offset,
			  center_x + offset,
			  center_y + offset)
   
   img = iproc.nega(new_image)
   return img
end
function preprocess.gcn(img)
   if img:dim() == 3 then
      return iproc.gcn(img)
   else
      local imgs = img
      for i = 1, imgs:size(1) do
	 imgs[i]:copy(iproc.gcn(imgs[i]))
      end
      return imgs
   end
end

function preprocess.reconstruct(img)
   img = centering(img)
   img = iproc.nega(img)
   return img
end

local function test()
   -- run with `qlua' command
   local tests = {"data/train/chaetognath_non_sagitta/4011.jpg",
		  "data/train/tunicate_salp/77923.jpg",
		  "data/train/copepod_cyclopoid_oithona/49184.jpg",
		  "data/train/protist_other/77117.jpg",
		  "data/train/tunicate_doliolid/63343.jpg" }
   for i = 1, #tests do
      local img = image.load(tests[i])
      image.save(string.format("figure/preprocess_before_%d.png", i), img)
      img = preprocess.reconstruct(img)
      image.save(string.format("figure/preprocess_after_%d.png", i), img)
   end
end
--test()

return preprocess
