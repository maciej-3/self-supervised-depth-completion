import torch
import time





from dataloaders.pointpix_loader import PointPixDataset, Arguments, display_rgb_or_depth
from model import DepthCompletionNet




########################################################################################################################
########################################################################################################################

args = Arguments()
args.cpu = True
args.data_folder = "/home/maciej/git/depth_compl/self-supervised-depth-completion/data_pp"
args.mode = "infer"
args.path2checkpoint = "/home/maciej/git/depth_compl/self-supervised-depth-completion/checkpoints/model_best.pth.tar"
args.layers = 34
args.input = "gd"
args.use_d = True
args.use_g = True
args.use_rgb = False
args.pretrained = False
args.num_workers = 0
#TODO change args class sos that you can print all the members using the print() function
print(args)


cuda = cuda = torch.cuda.is_available() and not args.cpu
if cuda:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("=> using '{}' for computation.".format(device))


checkpoint = torch.load(args.path2checkpoint, map_location=device)







########################################################################################################################
########################################################################################################################
def iterate(args, model, data_loader, optimiser=None, logger=None, epoch=1):
    # assert proper mode of working
    assert args.mode in ["train", "val", "eval", "test_prediction", "test_completion", "infer"], \
        "unsupported mode: {}".format(args.mode)

    if args.mode == 'train':
        model.train()
        # lr = helper.adjust_learning_rate(args.lr, optimiser, epoch)
    else:
        model.eval()
        lr = 0

    # start iteration
    for i, batch_data in enumerate(data_loader):
        batch_data = {
            key: val.to(device)
            for key, val in batch_data.items() if val is not None
        }
        # gt = batch_data[
        #     'gt'] if mode != 'test_prediction' and mode != 'test_completion' else None
        # data_time = time.time() - start

        # make the prediction
        start = time.time()
        print("\t=> starting forward pass ... ", end='')
        pred = model(batch_data)
        print("completed.")
        pred_time_stop = time.time() - start
        print("\t=> prediction time: {}".format(pred_time_stop))

        # display the prediction
        if cuda : pred = pred.to('cpu')
        display_rgb_or_depth(pred.detach().numpy())



########################################################################################################################
########################################################################################################################
def main():

    # instantiate a model
    print("=> creating model ... ", end='')
    model = DepthCompletionNet(args).to(device)
    print("completed.")

    # load their checkpoint
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> checkpoint state loaded.")

    # move the loaded checkpoint thingies to the proper device
    if cuda : model = torch.nn.DataParallel(model)


    # create my data loader
    print("=> creating data loaders ... ", end='')
    infer_dataset = PointPixDataset(args)
    infer_loader = torch.utils.data.DataLoader(
        infer_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True)  # set batch size to be 1 for validation
    print("completed.")
    print("\t==> data_loader size: {}".format(len(infer_loader)))


    # create a logger and results saver
    # TODO


    # nake a prediction while iterating
    print("=> starting iterating ...")
    iterate(args, model, infer_loader)
    print("=> finished iterating")






if __name__ == '__main__':
    main()