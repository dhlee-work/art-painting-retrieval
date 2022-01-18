import torch
from torchvision import transforms
from retrieval.models.aecls import AECLS

class aecls():
    def __init__(self, weight_path=''):
        self.data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(199),
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Load Model ...")
        self.model = AECLS()
        self.model.load_state_dict(torch.load(weight_path))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, img):
        _img = img.transpose(2, 0, 1)
        _img = torch.tensor(_img)
        _img = self.data_transforms(_img).to(self.device)
        if len(_img.shape) == 3:
            _img = torch.unsqueeze(_img, 0)
        out_rec = self.model.encode(_img)
        return out_rec.detach().cpu().detach().numpy()
