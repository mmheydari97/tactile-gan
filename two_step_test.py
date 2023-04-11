import argparse
import os
from test import *


def test_two_step(gen1, gen2, dataset, output_path, evaluation=True):
    accuracy = []
    jaccard = []
    dice = []

    if not os.path.exists(os.path.join(output_path, "out")):
        os.makedirs(os.path.join(output_path, "out"))
    if not os.path.exists(os.path.join(output_path, "sgt")):
        os.makedirs(os.path.join(output_path, "sgt"))
    if not os.path.exists(os.path.join(output_path, "elm")):
        os.makedirs(os.path.join(output_path, "elm"))
       

    for i, batch in enumerate(tqdm(dataset)):
        real_A, real_B = batch[0], batch[1]
        with torch.no_grad():
            intermediate = gen1(real_A.to(device))
            out = gen2(intermediate.to(device)).cpu()
        a = unnormalize(real_A[0])
        b = real_B[0]
        out = out[0]
        
        if evaluation:
            res = eval_pair(b, out)
            accuracy.append(res["accuracy"])
            dice.append(res["dice"])
            jaccard.append(res["jaccard"])

        
        b_img = visualize(b)
        out_img = visualize(out)
            
        out_img.save(os.path.join(output_path, "out", f"{i+1}.png"))
        concat_images(ToPILImage()(a), b_img, out_img).save(os.path.join(output_path, "sgt", f"{i+1}.png"))

        b_elements = concat_images(ToPILImage()(b[0]), ToPILImage()(b[1]), ToPILImage()(b[2]))
        out_elements = concat_images(ToPILImage()(out[0]), ToPILImage()(out[1]), ToPILImage()(out[2]))
        concat_images(b_elements,out_elements, mode="v").save(os.path.join(output_path, "elm", f"{i+1}.png"))
    return accuracy, dice, jaccard

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1_dir", default="t1_2d_per")
    parser.add_argument("--s2_dir", default="t2_2d_per")
    parser.add_argument("--data", default="data_plot_3")
    opt = parser.parse_args()

    opt_path_1 = os.path.join(os.getcwd(),"models", opt.s1_dir.split("/")[-1], "params.txt")
    opt1 = load_opt(opt_path_1)
    
    opt_path_2 = os.path.join(os.getcwd(),"models", opt.s2_dir.split("/")[-1], "params.txt")
    opt2 = load_opt(opt_path_2)

    device = torch.device("cuda:0")

    model_path_1 = os.path.join(os.getcwd(),"models",opt1.folder_save,"final_model.pth")
    model_path_2 = os.path.join(os.getcwd(),"models",opt2.folder_save,"final_model.pth")
    
    gen1 = load_model(model_path_1,opt1,device)
    gen2 = load_model(model_path_2,opt2,device)

    photo_path_test= os.path.join(os.getcwd(),opt.data,"test","source")
    dataset = load_data(photo_path_test,opt2, shuffle=False)

    output_path = os.path.join(os.getcwd(),"Outputs",f"{opt.s1_dir}+{opt.s2_dir}")
    mkdir(output_path)
    
    accuracy, dice, jaccard = test_two_step(gen1, gen2, dataset, output_path, evaluation=True)
    if len(accuracy)>0:
        print_evaluation(accuracy, dice, jaccard, output_path)


