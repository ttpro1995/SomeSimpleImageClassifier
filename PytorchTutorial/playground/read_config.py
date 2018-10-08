import configparser

if __name__ == "__main__":
    print("meow")
    config = configparser.ConfigParser()
    config.read("config.ini")
    print(config)
    print(config["MNIST"]["dataset"])
    print(config["MNIST"])