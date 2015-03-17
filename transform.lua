require 'image'
require './util'

local iproc = require './iproc'
local preprocess = require './preprocess'
local transform = {}

-- data augmentation methods

local DA_FACTOR = 0.125
local DA_FACTOR_MIN = 0.1
local DA_FACTOR_MAX = 0.1666

local function calc_crops(size, factor)
   local off = math.floor(factor * size)
   local off5 = math.floor(off * 0.5)
   local w = off * 2 + size - 1
   local crop_p4s = {
      -- zoom
      { off, off, w - off, off, w - off, w - off, off, w - off },
      { 0, 0, w, 0, w, w, 0, w},
      { off*2, off*2, w - off * 2, off * 2, w - off * 2, w - off * 2, off * 2, w - off * 2 },
      
      -- perspective crop
      
      -- zoomout
      { off, 0, w - off, 0, w - off, w, off, w},
      { 0, off, w, off, w, w - off, 0, w - off},
      {off, off, w - off, off, w, w - off, 0, w - off},
      {0, off, w, off, w - off, w - off, off, w - off},
      {off, 0, w - off, off, w - off, w - off, off, w},
      {off, off, w - off, 0, w - off, w, off, w - off},
      {off, 0, w - off, off, w - off, w - off, off, w},

      -- zoomin
      { off*2, off, w - off * 2, off, w - off * 2, w - off, off * 2, w - off},
      { off, off*2, w - off, off * 2, w - off, w - off * 2, off, w - off * 2},
      {off + off, off + off, w - off - off, off + off, w - off, w - off - off, off, w - off - off},
      {off, off + off, w - off, off + off, w - off - off, w - off - off, off + off, w - off - off},
      {off + off, off, w - off - off, off + off, w - off - off, w - off - off, off + off, w - off},
      {off + off, off + off, w - off - off, off, w - off - off, w - off, off + off, w - off - off},
      {off + off, off, w - off - off, off + off, w - off - off, w - off - off, off + off, w - off},
   }
   return crop_p4s
end

local function generate_fixed_accurate(src, size)
   size = size or src:size(2)
   local org_size = src:size(2)
   local off = math.floor(DA_FACTOR * org_size)
   local base = {}
   local images = {}
   local angles = {
      0,
      math.pi / 4 * 1,
      math.pi / 4 * 2,
      math.pi / 4 * 3,
      math.pi / 4 * 4,
      math.pi / 4 * 5,
      math.pi / 4 * 6,
      math.pi / 4 * 7
   }
   src = iproc.zero_padding(src, off)
   for i = 1, #angles do
      local a
      if angles[i] == 0 then
	 a = src
      else
	 a = iproc.rotate(src, angles[i])
      end
      table.insert(base, a)
      table.insert(base, image.hflip(a))
   end
   for i = 1, #base do
      local crop_p4s = calc_crops(org_size, DA_FACTOR)
      for j = 1, #crop_p4s do
	 local p4 = crop_p4s[j]
	 table.insert(images,
		      iproc.perspective_crop(base[i],
					     p4[1], p4[2],
					     p4[3], p4[4],
					     p4[5], p4[6],
					     p4[7], p4[8],
					     size, size)
	 )
      end
   end
   
   return images
end
local function generate_fixed(src, size)
   local size = size or src:size(2)
   local org_size = src:size(2)
   local off = math.floor(DA_FACTOR * org_size)
   local base = {}
   local images = {}
   local angles = {
      0,
      math.pi / 4 * 1,
      math.pi / 4 * 2,
      math.pi / 4 * 3,
      math.pi / 4 * 4,
      math.pi / 4 * 5,
      math.pi / 4 * 6,
      math.pi / 4 * 7
   }
   src = iproc.zero_padding(src, off)
   
   for i = 1, #angles do
      local a
      if angles[i] == 0 then
	 a = src
      else
	 a = iproc.rotate(src, angles[i])
      end
      table.insert(base, a)
      table.insert(base, image.hflip(a))
   end
   for i = 1, #base do
      local crop_p4s = calc_crops(org_size, DA_FACTOR)
      for j = 1, 3 do
	 local p4 = crop_p4s[j]
	 table.insert(images,
		      iproc.perspective_crop(base[i],
					     p4[1], p4[2],
					     p4[3], p4[4],
					     p4[5], p4[6],
					     p4[7], p4[8],
					     size, size))
      end
   end
   return images
end
function transform.augment(x, size)
   -- jitter for validation
   size = size or x:size(2)
   local images = generate_fixed(x, size)
   local new_x = torch.Tensor(#images, 1, size, size)
   for i = 1, #images do
      new_x[i]:copy(images[i])
   end
   return new_x
end
function transform.augment_accurate(x, size)
   -- jitter for prediction
   size = size or x:size(2)
   local images = generate_fixed_accurate(x, size)
   local new_x = torch.Tensor(#images, 1, size, size)
   for i = 1, #images do
      new_x[i]:copy(images[i])
   end
   return new_x
end
function transform.random(x, size, factor)
   -- jitter for training
   size = size or x:size(2)
   factor = factor or torch.uniform(DA_FACTOR_MIN, DA_FACTOR_MAX)
   local off = math.floor(factor * x:size(2))
   local rot = torch.uniform(0.0, 2.0 * math.pi)
   local shift_x = torch.uniform(-off, off)
   local shift_y = torch.uniform(-off, off)
   local crop_p4s = calc_crops(x:size(2), factor)
   local p4 = crop_p4s[torch.random(1, #crop_p4s)]
   local contrast = torch.uniform(0.8, 1.2)   

   x = iproc.scale_contrast(x, contrast)
   x = iproc.zero_padding(x, off)
   x = image.translate(x, shift_x, shift_y)
   if torch.uniform() > 0.5 then
      x = image.hflip(x)
   end
   x = iproc.rotate(x, rot)
   x = iproc.perspective_crop(x,
			      p4[1], p4[2],
			      p4[3], p4[4],
			      p4[5], p4[6],
			      p4[7], p4[8],
			      size, size)

   return x
end
local function test_crop()
   local src = image.load("grid.png")
   local org_size = src:size(2)
   local off = DA_FACTOR * src:size(2)
   local crop_p4s = calc_crops(org_size, DA_FACTOR)
   local imgs = torch.Tensor(#crop_p4s, 1, IMG_SIZE, IMG_SIZE)
   src = iproc.zero_padding(src, off)   
   for i = 1, imgs:size(1) do
      local p4 = crop_p4s[i]
      imgs[i]:copy(
	 iproc.perspective_crop(src,
				p4[1], p4[2],
				p4[3], p4[4],
				p4[5], p4[6],
				p4[7], p4[8],
				IMG_SIZE, IMG_SIZE)
		  )
   end
   imgs = preprocess.gcn(imgs)
   save_images(imgs, imgs:size(1), "jitter_crop.png")
end

local function test_random()
   local target = 2033
   local src = torch.load("data/train_x.t7")[target]
   local grid = image.load("figure/grid.png")
   local imgs = torch.Tensor(64, 1, 48, 48)
   local imgs2 = torch.Tensor(64, 1, 48, 48)

   torch.manualSeed(1)
   for i = 1, imgs:size(1) do
      imgs[i]:copy(transform.random(src, 48))
   end
   imgs = preprocess.gcn(imgs)
   save_images(imgs, imgs:size(1), "figure/random_transform.png")
   
   torch.manualSeed(1)
   for i = 1, imgs2:size(1) do
      imgs2[i]:copy(transform.random(grid, 48))
   end
   imgs2 = preprocess.gcn(imgs2)
   save_images(imgs2, imgs2:size(1), "figure/random_transform_grid.png")
end
local function test_accurate()
   local target = 2033
   local src = torch.load("data/train_x.t7")[target]
   local imgs = transform.augment(src, 48)
   imgs = preprocess.gcn(imgs)
   save_images(imgs, imgs:size(1), "figure/augment_accurate.png")
end

--test_crop()
--test_random()
--test_accurate()

return transform
