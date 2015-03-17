local settings = require './settings'

function load_classes()
   local classes = {}
   local class2id = {}
   local id2class = {}
   local class_file = string.format("%s/classes.txt", settings.data_dir)
   local fp = io.open(class_file, "r")
   local i = 1
   for line in fp:lines() do
      id2class[i] = line
      class2id[line] = i
      i = i + 1
   end
   fp.close()
   return class2id, id2class
end
function slice(x, n)
   local new_x = {}
   for i = 1, n do
      new_x[i] = x[i]
   end
   return new_x
end

function logloss(y, p)
   return -1 * y:clone():cmul(torch.log(p + 1.0e-15)):sum()
end

function save_images(x, n, file)
   file = file or "./out.png"
   local input = x:narrow(1, 1, n)
   local view = image.toDisplayTensor({input = input,
				       padding = 2,
				       nrow = 8,
				       symmetric = true})
   image.save(file, view)
end
