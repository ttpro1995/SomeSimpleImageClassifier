def get_img_id(in_str):
    ### '/data/cv_data/minist/mnistasjpg/wraptest/testSet/img_1.jpg'
    return in_str.split(".")[0].split("_")[-1]


if __name__ == "__main__":
    sample = '/data/cv_data/minist/mnistasjpg/wraptest/testSet/img_1.jpg'
    print(get_img_id(sample))
