import argparse
import traceback

from QEfficient import QEFFAutoModelForImageTextToText


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-4-E2B-it")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    model = QEFFAutoModelForImageTextToText.from_pretrained(args.model)
    print("loaded", type(model), type(model.model), flush=True)

    export_path = model.export()
    print("export_path", export_path, flush=True)

    if args.compile:
        qpc_path = model.compile(num_cores=16, mxfp6_matmul=True)
        print("qpc_path", qpc_path, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
