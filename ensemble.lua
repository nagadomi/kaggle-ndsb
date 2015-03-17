require 'cutorch'
require 'cunn'
require './util'

local settings = require './settings'

local function load_submission_file(file)
   local fp = io.open(file, "r")
   local preds = {}
   local head = false
   for line in fp:lines() do
      if head then
	 local cols = utils.split(line, ",")
	 local vec = {}
	 for i = 1, #cols - 1 do
	    vec[i] = tonumber(cols[i + 1])
	 end
	 table.insert(preds, vec)
      else
	 head = true
      end
   end
   fp:close()
   
   return torch.Tensor(preds)
end
local function average_predictions(files)
   local preds = {}
   
   for i = 1, #files do
      preds[i] = load_submission_file(files[i])
   end
   
   local z = torch.Tensor():resizeAs(preds[1]):zero()
   for i = 1, #preds do
      z:add(preds[i])
   end
   z:div(#preds)
   
   return z
end
local function load_filenames()
   local fp = io.open(string.format("%s/test.txt", settings.data_dir))
   local filenames = {}
   local i = 1
   for line in fp:lines() do
      local names = utils.split(line, "/")
      filenames[i] = names[#names]
      i = i + 1
   end
   return filenames
end

local function ensemble()
   local filenames = load_filenames()
   local class2id, id2class = load_classes()
   local submission_48 = {
      "models/submission_48x_stratified_101.txt",
      "models/submission_48x_stratified_102.txt",
      "models/submission_48x_stratified_103.txt",
      "models/submission_48x_stratified_104.txt",
      "models/submission_48x_stratified_105.txt",
      "models/submission_48x_stratified_106.txt",
      "models/submission_48x_stratified_107.txt",
      "models/submission_48x_stratified_108.txt"
   }
   local submission_72 = {
      "models/submission_72x_stratified_101.txt",
      "models/submission_72x_stratified_102.txt",
      "models/submission_72x_stratified_103.txt",
      "models/submission_72x_stratified_104.txt",
      "models/submission_72x_stratified_105.txt",
      "models/submission_72x_stratified_106.txt",
      "models/submission_72x_stratified_107.txt",
      "models/submission_72x_stratified_108.txt"
   }
   local submission_96 = {
      "models/submission_96x_stratified_101.txt",
      "models/submission_96x_stratified_102.txt",
      "models/submission_96x_stratified_103.txt",
      "models/submission_96x_stratified_104.txt",
      "models/submission_96x_stratified_105.txt",
      "models/submission_96x_stratified_106.txt",
      "models/submission_96x_stratified_107.txt",
      "models/submission_96x_stratified_108.txt",
   }
   local z48 = average_predictions(submission_48)
   local z72 = average_predictions(submission_72)
   local z96 = average_predictions(submission_96)
   
   -- best param: w48=0.22928008437157, w72=0.34947738051414, w96=0.42124256491661, bias=9.8288448294625e-06
   -- found this param with appendix/cv_find_param.lua
   
   local z = z48:mul(0.22928008437157) + z72:mul(0.34947738051414) + z96:mul(0.42124256491661)
   z:add(9.8288448294625e-06)
   
   io.stdout:write("image")
   for i = 1, #id2class do
      io.stdout:write(",")
      io.stdout:write(id2class[i])
   end
   io.stdout:write("\n")

   for i = 1, z:size(1) do
      io.stdout:write(filenames[i] .. ",")
      z[i]:div(z[i]:sum())
      for j = 1, z:size(2) do
	 local p = 0.0
	 if j ~= 1 then
	    io.stdout:write(",")
	 end
	 io.stdout:write(z[i][j])
      end
      io.stdout:write("\n")
   end
end
ensemble()
