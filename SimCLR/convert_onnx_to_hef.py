import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(
        description="Convert ONNX model to HEF format using Hailo Data Flow Compiler")
    parser.add_argument("--ckpt", required=True, help="Chemin vers le fichier ONNX")
    parser.add_argument("--hw_arch", default="hailo8l", help="Architecture hardware (par défaut: hailo8l)")
    parser.add_argument("--calib_path", required=True, help="Chemin vers le dossier d'images de calibration")
    parser.add_argument("--classes", type=int, required=True, help="Nombre de classes (par exemple, 1 pour extraction d'embeddings)")
    parser.add_argument("--performance", action="store_true", help="Activer le mode performance")
    parser.add_argument("--output", required=True, help="Nom du fichier HEF de sortie (ex: custom_encoder.hef)")
    args = parser.parse_args()

    cmd = [
        "hailomz", "compile", "custom_encoder",
        "--ckpt", args.ckpt,
        "--hw-arch", args.hw_arch,
        #"--calib-path", args.calib_path,
        "--classes", str(args.classes)
    ]

    if args.performance:
        cmd.append("--performance")

    print("Commande lancée :", " ".join(cmd))
    subprocess.run(cmd, check=True)

    subprocess.run(["mv", "custom_encoder.hef", args.output], check=True)
    print("Conversion terminée. Le fichier HEF est :", args.output)

if __name__ == "__main__":
    main()
