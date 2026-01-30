
from losses.mylosses import TransMemNetLoss,GroupDROLoss,VRExLoss,IRMLoss


def get_loss(backbone,lambda_sparse=0.0,label_smoothing=0.0):
    
    if backbone =='irm':
        return IRMLoss(lambda_irm=0.3)
        
    elif backbone =='vrex':
        return VRExLoss(lambda_vrex=0.3)
    elif backbone =='dro':
        return GroupDROLoss(eta=0.1)
    else:
        
        return TransMemNetLoss(lambda_sparse=lambda_sparse)
        
   

