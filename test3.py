import argparse


def main():
    parser = argparse.ArgumentParser("Data Load")
    parser.add_argument("--wrist", action="store_true")
    # parser.add_argument("--side", dest="side", action="store_true")
    parser.add_argument("--no-side", dest="side", action="store_false")
    parser.set_defaults(side=True)
    args = parser.parse_args()

    print(args.wrist)
    print(args.side)


if __name__ == "__main__":
    main()
