def get_img_id(in_str):
    """
    parse the path to get name
    :param in_str:
    :return:
    """
    # '/data/cv_data/ai/testset/public_test/1000082.jpg'
    return in_str.split(".")[0].split("/")[-1]


if __name__ == "__main__":
    instr = '/data/cv_data/ai/testset/public_test/1000082.jpg'
    print(get_img_id(instr))
