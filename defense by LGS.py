from local_gradients_smoothing.configs import Configuration
from local_gradients_smoothing.lgs import LocalGradientsSmoothing
from torchvision.transforms import ToTensor, ToPILImage  

#print(p_tensor_batch.shape)

to_pil = transforms.ToPILImage()
#p_tensor_batch = to_pil(p_tensor_batch[0])
#print(p_tensor.shape)
#img = Image.fromarray(img)

def lgs(img):
  
        cfg = Configuration()
        img = to_pil(img[0])
        loc_grad_smooth = LocalGradientsSmoothing(**cfg.get('DEFAULT'))
        grad_mask = loc_grad_smooth(img).squeeze(0)

        grad_mask = grad_mask.repeat((3, 1, 1))
        img_t = ToTensor()(img)
        collage_t = img_t * (1 - grad_mask)
        #collage = ToPILImage()(grad_mask)
        #collage.save("/Data3/mj23/output_image.png")
        
        #print(collage_t.shape)
        
        return collage_t, grad_mask
  
p_tensor_batch, _ = lgs(p_tensor_batch)



p_tensor_batch = p_tensor_batch.cuda() 
p_tensor_batch=p_tensor_batch.unsqueeze(dim=0)
