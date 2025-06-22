import glob
import numpy as np
from scipy import stats
from PIL import Image
def get_3D_Dose_dif():
    # 定义预测图像和真实图像的路径
    pre_mamba_path = r'D:\mamba\mamba_dose\Test_Output\ddpm-unet\hx-psdm-fusion-struc-mamba-unet-all-image'
    pre_trans_path = r'D:\ddpm-dose\Test_Output\ddpm-unet\hxdose4-all-image'
    gt_path = r'D:\mamba\mamba_dose\Test_Output\ddpm-unet\gt'
    pre_mamba_pt = sorted(glob.glob(pre_mamba_path + '/*/*.png'))
    pre_trans_pt = sorted(glob.glob(pre_trans_path + '/*/*.png'))
    gt_pt = sorted(glob.glob(gt_path + '/*/*.png'))
    count = 0
    differences = []
    differences1 = []
    result = []
    min = 999
    for i, j,k in zip(pre_mamba_pt, gt_pt,pre_trans_pt):
        im1 = Image.open(i)
        im2 = Image.open(j)
        im3 = Image.open(k)
        im1 = im1.convert('L')
        im2 = im2.convert('L')
        im3 = im3.convert('L')

        dif1 = np.mean(np.abs(np.array(im1) / 255 * 70 - np.array(im2) / 255 * 70))
        dif2 = np.mean(np.abs(np.array(im3) / 255 * 70 - np.array(im2) / 255 * 70))
        count = count + dif1
        differences.append(np.mean(np.abs(np.array(im1) / 255 * 70 - np.array(im2) / 255 * 70)))
        differences1.append(np.mean(np.abs(np.array(im3) / 255 * 70 - np.array(im2) / 255 * 70)))
        if (dif1 < min):
            min = dif1
            min_path = i
    sample1 = np.asarray(differences)
    sample2 = np.asarray(differences1)
    diff=sample1-sample2
    lenth=len(diff)
    negative_count = sum(1 for num in diff if num > 0)
    print(negative_count/lenth)
    r = stats.ttest_ind(sample1, sample2)
    # result.append(r[1])
    print("statistic:", r.__getattribute__("statistic"))
    print("pvalue:", r.__getattribute__("pvalue"))

    # print(np.mean(result))
    avg_diff = count / len(pre_mamba_pt)
    std_dev = np.std(differences)
    var_dev = np.var(differences)
    print(min, min_path)
    print("平均差异：", avg_diff)
    print("标准差：", std_dev)
    print("方差:", var_dev)
def get_dvh():
    pre_mamba_path = r'D:\mamba\mamba_dose\Test_Output\ddpm-unet\hx-psdm-fusion-struc-mamba-unet-all-image'
    pre_trans_path = r'D:\ddpm-dose\Test_Output\ddpm-unet\hxdose4-all-image'
    gt_path = r'D:\mamba\mamba_dose\Test_Output\ddpm-unet\gt'
    pre_mamba_pt = sorted(glob.glob(pre_mamba_path + '/*/*.png'))
    pre_trans_pt = sorted(glob.glob(pre_trans_path + '/*/*.png'))
    gt_pt = sorted(glob.glob(gt_path + '/*/*.png'))
    count=0
    list_DVH_dif = []
    list_DVH_dif1 = []
    result=[]
    differences = []
    min=999
    min1 = 999
    for i, j,k in zip(pre_mamba_pt, gt_pt,pre_trans_pt):

        output = {}
        output1 = {}
        output2={}
        im1 = Image.open(i)
        im2 = Image.open(j)
        im3=Image.open(k)
        im1 = im1.convert('L')
        im2 = im2.convert('L')
        im3 = im3.convert('L')
        _roi_dose = np.array(im1)/255
        # D1
        output['D1'] = np.percentile(_roi_dose, 99)
        # D95
        output['D95'] = np.percentile(_roi_dose, 5)
        # D99
        output['D99'] = np.percentile(_roi_dose, 1)

        _roi_dose1 = np.array(im2)/255
        # D1
        output1['D1'] = np.percentile(_roi_dose1, 99)
        # D95
        output1['D95'] = np.percentile(_roi_dose1, 5)
        # D99
        output1['D99'] = np.percentile(_roi_dose1, 1)

        _roi_dose2 = np.array(im3)/255
        # D1
        output2['D1'] = np.percentile(_roi_dose2, 99)
        # D95
        output2['D95'] = np.percentile(_roi_dose2, 5)
        # D99
        output2['D99'] = np.percentile(_roi_dose2, 1)
        for metric in output.keys():
            list_DVH_dif.append(np.mean(np.abs(output[metric] - output1[metric])))
            if(np.mean(np.abs(output[metric] - output1[metric]))<min):
                min=np.mean(np.abs(output[metric] - output1[metric]))
                min_path=i
        for metric in output2.keys():
            list_DVH_dif1.append(np.mean(np.abs(output2[metric] - output1[metric])))
            if(np.mean(np.abs(output2[metric] - output1[metric]))<min):
                min1=np.mean(np.abs(output2[metric] - output1[metric]))
                min_path1=k
    sample1 = np.asarray(list_DVH_dif)
    sample2 = np.asarray(list_DVH_dif1)
    r = stats.ttest_ind(sample1, sample2)
    # result.append(r[1])
    print("statistic:", r.__getattribute__("statistic"))
    print("pvalue:", r.__getattribute__("pvalue"))
    print(np.mean(list_DVH_dif)*70)
    print(np.std(list_DVH_dif)*70)
def HI():
    pre_path = r'D:\mamba\mamba_dose\Test_Output\ddpm-unet\hx-psdm-fusion-struc-mamba-unet-all-image'
    gt_path = r'D:\mamba\mamba_dose\Test_Output\ddpm-unet\gt'
    pre_pt = sorted(glob.glob(pre_path + '/*/*.png'))
    gt_pt = sorted(glob.glob(gt_path + '/*/*.png'))
    count = 0
    differences = []
    min = 999

    for i, j in zip(pre_pt, gt_pt):
        im1 = Image.open(i)
        im2 = Image.open(j)

        im1 = np.array(im1.convert('L'))/ 255
        im2 = np.array(im2.convert('L'))/ 255
        predicted_dose_flat = im1.flatten()
        true_dose_flat = im2.flatten()


        # 计算 HI 指数
        def heterogeneity_index(image_array,path):
            # 计算像素值的均值
            mean_value = np.mean(image_array)
            # 计算像素值的标准差
            std_deviation = np.std(image_array)
            # 计算 HI 指数
            if mean_value < 1e-10:
                print(path)
                mean_value = 1e-10
            heterogeneity_index = std_deviation / mean_value
            return heterogeneity_index

        # 计算预测的剂量图像和真实的剂量图像的 HI 指数
        # print(i)
        hi_predicted = heterogeneity_index(predicted_dose_flat,path=i)
        hi_true = heterogeneity_index(true_dose_flat,path=j)
        delta_hi = np.abs(hi_predicted - hi_true)
        count = count + delta_hi
        differences.append(delta_hi)
    avg_diff = count / len(pre_pt)
    std_dev = np.var(differences)
    print("平均差异：", avg_diff)
    print("标准差：", std_dev)
