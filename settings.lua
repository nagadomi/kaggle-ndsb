require 'torch'
require 'cutorch'

-- global settings

if package.preload.settings then
   return package.preload.settings
end

-- default tensor type
torch.setdefaulttensortype('torch.FloatTensor')

-- CPU cores
torch.setnumthreads(4)

local settings = {}

-- data directory
settings.data_dir = "./data"
-- trained model directory
settings.model_dir = "./models"

-- specifiy the data directory
settings.image_size = 48

-- number of classes
settings.classes = 121

local cmd = torch.CmdLine()
cmd:text()
cmd:text("Kaggle-BOWL")
cmd:text("Options:")
cmd:option("-seed", 11, 'fixed input seed')
cmd:option("-split", "stratified", 'split method (unrestricted|stratified)')
cmd:option("-model", 48, 'model (48 | 72 | 96)')
cmd:option("-predict", "fast", 'predict method (fast| accuracy)')
cmd:option("-data_dir", "./data", 'data directory')
cmd:option("-model_dir", "./models", 'model directory')

local opt = cmd:parse(arg)

for k, v in pairs(opt) do
   settings[k] = v
end
settings.image_size = settings.model

if not (settings.split == "stratified" or settings.split == "unrestricted") then
   error("undefined split method :" .. settings.split)
end
if settings.model == 48 then
   settings.create_model =  require './cnn_48x48'
elseif settings.model == 72 then
   settings.create_model =  require './cnn_72x72'
elseif settings.model == 96 then
   settings.create_model =  require './cnn_96x96'
else
   error("undefined model " .. opt.model)
end

settings.model_file = string.format("%s/cnn_%dx_%s_%d.t7",
				    settings.model_dir,
				    settings.model,
				    settings.split,
				    settings.seed)
settings.submission_file = string.format("%s/submission_%dx_%s_%d.txt",
					settings.model_dir,
					settings.model,
					settings.split,
					settings.seed)

function settings.set_seed()
   torch.manualSeed(settings.seed)
   cutorch.manualSeed(settings.seed)
end

package.preload.settings = settings

return settings
