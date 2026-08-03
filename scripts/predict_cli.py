import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.inference import PharmaSentinelPredictor

def main():
    parser = argparse.ArgumentParser(description="PharmaSentinel CLI Multi-Drug Risk Predictor")
    parser.add_argument("--drugs", nargs="+", help="1 to 4 Drug Names or STITCH IDs (e.g. Warfarin, Aspirin, Metformin)")
    args = parser.parse_args()

    predictor = PharmaSentinelPredictor()
    drugs_input = args.drugs if args.drugs and len(args.drugs) > 0 else ["Warfarin"]

    print("\n==========================================================")
    print("  PharmaSentinel - Clinical Regimen & FDA Safety Predictor ")
    print("==========================================================\n")

    res = predictor.predict_regimen(drugs_input)

    print(f"  Regimen Mode  : {res['mode'].upper()} ({res['num_drugs']} Drugs)")
    print(f"  Drugs Regimen : " + "  +  ".join([d['display'] for d in res['drugs']]))
    print(f"  Overall Risk  : {res['risk_level']} ({res['overall_risk_score']}%)")

    if res.get("has_blackbox"):
        print("\n  =================== FDA BLACK BOX WARNING ===================")
        print(f"  {res['blackbox_warning']}")
        print("  =============================================================\n")

    if res['num_drugs'] > 1 and "driver_pair" in res:
        driver = res['driver_pair']
        print(f"  Driver Pair   : {driver['pair_display']} (Highest Hazard: {driver['pair_risk_score']}%)")

    print("\nTop Adverse Event Warnings:")
    print("----------------------------------------------------------")
    if res['num_drugs'] == 1:
        for item in res['isolated_side_effects']:
            print(f"  • {item['side_effect']:<40} | [{item['severity']}] ({item.get('probability', 72.0):.1f}%)")
    else:
        for item in res['top_predicted_side_effects']:
            print(f"  • {item['side_effect']:<40} | Prob: {item['probability']:>5.1f}% | [{item['severity']}]")
    print("----------------------------------------------------------\n")

if __name__ == "__main__":
    main()
