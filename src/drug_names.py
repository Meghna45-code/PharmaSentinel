import os
import json
import urllib.request
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
NAMES_MAP_FILE = os.path.join(DATA_DIR, "drug_names_map.json")

# Common fallback dictionary for STITCH CIDs to Generic Drug Names
COMMON_DRUG_NAMES = {
    "CID000000085": "Carnitine",
    "CID000000119": "gamma-Aminobutyric acid (GABA)",
    "CID000000143": "Glutathione",
    "CID000000158": "Histamine",
    "CID000000159": "Histidine",
    "CID000000191": "Melatonin",
    "CID000000206": "Nicotinic acid (Niacin)",
    "CID000000214": "Penicillamine",
    "CID000000271": "Pyridoxine (Vitamin B6)",
    "CID000000298": "Serotonin",
    "CID000000444": "Thiamine (Vitamin B1)",
    "CID000000450": "Thyroxine (T4)",
    "CID000000453": "Tocopherol (Vitamin E)",
    "CID000000564": "Acetylcysteine",
    "CID000000581": "Allopurinol",
    "CID000000596": "Aminophylline",
    "CID000000598": "Amitriptyline",
    "CID000000772": "Ampicillin",
    "CID000000815": "Aspirin (Acetylsalicylic acid)",
    "CID000000838": "Atropine",
    "CID000000853": "Baclofen",
    "CID000000861": "Bethanechol",
    "CID000000937": "Caffeine",
    "CID000000942": "Calcitriol (Vitamin D3)",
    "CID000001046": "Chloramphenicol",
    "CID000001065": "Chloroquine",
    "CID000001071": "Chlorpromazine",
    "CID000001117": "Cimetidine",
    "CID000001134": "Ciprofloxacin",
    "CID000001206": "Clonidine",
    "CID000001302": "Cyclophosphamide",
    "CID000001546": "Dexamethasone",
    "CID000001690": "Diazepam (Valium)",
    "CID000001775": "Digoxin",
    "CID000001971": "Dopamine",
    "CID000001972": "Doxorubicin",
    "CID000001978": "Doxycycline",
    "CID000001983": "Enalapril",
    "CID000001986": "Epinephrine (Adrenaline)",
    "CID000002019": "Erythromycin",
    "CID000002022": "Estradiol",
    "CID000002083": "Famotidine",
    "CID000002088": "Felodipine",
    "CID000002092": "Fenofibrate",
    "CID000002099": "Fentanyl",
    "CID000002118": "Fluconazole",
    "CID000002130": "Fluorouracil (5-FU)",
    "CID000002141": "Fluoxetine (Prozac)",
    "CID000002142": "Fluoxymesterone",
    "CID000002153": "Folic acid",
    "CID000002156": "Foscarnet",
    "CID000002160": "Fosinopril",
    "CID000002162": "Furosemide (Lasix)",
    "CID000002170": "Gabapentin",
    "CID000002171": "Galantamine",
    "CID000002173": "Ampicillin / Amoxicillin",
    "CID000002177": "Gemfibrozil",
    "CID000002182": "Gentamicin",
    "CID000002187": "Glibenclamide (Glyburide)",
    "CID000002215": "Glipizide",
    "CID000002232": "Haloperidol",
    "CID000002244": "Heparin",
    "CID000002249": "Hydrochlorothiazide",
    "CID000002250": "Hydrocortisone",
    "CID000002265": "Ibuprofen",
    "CID000002266": "Idarubicin",
    "CID000002267": "Ifosfamide",
    "CID000002269": "Imipenem",
    "CID000002274": "Imipramine",
    "CID000002283": "Indomethacin",
    "CID000002284": "Insulin",
    "CID000002308": "Isoniazid",
    "CID000002311": "Isosorbide mononitrate",
    "CID000002349": "Ketoconazole",
    "CID000002369": "Labetalol",
    "CID000002370": "Lactulose",
    "CID000002375": "Lamotrigine",
    "CID000002405": "Levodopa (L-DOPA)",
    "CID000002435": "Lidocaine",
    "CID000002462": "Lisinopril",
    "CID000002471": "Lithium",
    "CID000002474": "Loperamide (Imodium)",
    "CID000002476": "Lorazepam (Ativan)",
    "CID000002477": "Losartan",
    "CID000002478": "Lovastatin",
    "CID000002487": "Magnesium sulfate",
    "CID000002512": "Meloxicam",
    "CID000002519": "Mercaptopurine",
    "CID000002520": "Meropenem",
    "CID000002522": "Mesalamine",
    "CID000002524": "Metformin",
    "CID000002541": "Methotrexate",
    "CID000002550": "Methyldopa",
    "CID000002551": "Methylprednisolone",
    "CID000002554": "Metoclopramide",
    "CID000002576": "Metoprolol",
    "CID000002578": "Metronidazole (Flagyl)",
    "CID000002585": "Midazolam",
    "CID000002609": "Morphine",
    "CID000002610": "Mupirocin",
    "CID000002617": "Mycophenolate mofetil",
    "CID000002622": "Naloxone (Narcan)",
    "CID000002631": "Naproxen (Aleve)",
    "CID000002637": "Neostigmine",
    "CID000002646": "Nifedipine",
    "CID000002650": "Nitroglycerin",
    "CID000002656": "Nitroprusside",
    "CID000002658": "Norepinephrine",
    "CID000002662": "Nystatin",
    "CID000002666": "Omeprazole (Prilosec)",
    "CID000002673": "Ondansetron (Zofran)",
    "CID000002675": "Oseltamivir (Tamiflu)",
    "CID000002676": "Oxacillin",
    "CID000002678": "Oxazepam",
    "CID000002708": "Paclitaxel (Taxol)",
    "CID000002712": "Paracetamol (Acetaminophen)",
    "CID000002713": "Paroxetine (Paxil)",
    "CID000002719": "Penicillin V",
    "CID000002720": "Pentobarbital",
    "CID000002725": "Perphenazine",
    "CID000002726": "Pethidine (Meperidine)",
    "CID000002727": "Phenobarbital",
    "CID000002732": "Phentolamine",
    "CID000002733": "Phenylephrine",
    "CID000002749": "Phenytoin (Dilantin)",
    "CID000002756": "Pilocarpine",
    "CID000002764": "Piroxicam",
    "CID000002771": "Pravastatin",
    "CID000002786": "Prednisolone",
    "CID000002800": "Prochlorperazine",
    "CID000002801": "Progesterone",
    "CID000002802": "Promethazine",
    "CID000002803": "Propofol",
    "CID000002806": "Propranolol",
    "CID000002812": "Propylthiouracil",
    "CID000002818": "Pseudoephedrine",
    "CID000002891": "Quetiapine (Seroquel)",
    "CID000002895": "Quinidine",
    "CID000002907": "Ramipril",
    "CID000002909": "Ranitidine (Zantac)",
    "CID000002949": "Rifampicin (Rifampin)",
    "CID000002951": "Risperidone (Risperdal)",
    "CID000002955": "Ritonavir",
    "CID000002973": "Salbutamol (Albuterol)",
    "CID000002978": "Salicylic acid",
    "CID000002995": "Scopolamine",
    "CID000003003": "Sertraline (Zoloft)",
    "CID000003007": "Sildenafil (Viagra)",
    "CID000003008": "Simvastatin (Zocor)",
    "CID000003016": "Sodium bicarbonate",
    "CID000003032": "Spironolactone",
    "CID000003040": "Streptomycin",
    "CID000003042": "Succinylcholine",
    "CID000003043": "Sucralfate",
    "CID000003059": "Sulfamethoxazole",
    "CID000003062": "Sulfasalazine",
    "CID000003066": "Sumatriptan (Imitrex)",
    "CID000003075": "Tacrolimus",
    "CID000003108": "Tamoxifen",
    "CID000003114": "Tetracycline",
    "CID000003117": "Theophylline",
    "CID000003121": "Thiopental",
    "CID000003143": "Timolol",
    "CID000003148": "Tobramycin",
    "CID000003152": "Tolbutamide",
    "CID000003154": "Tramadol (Ultram)",
    "CID000003157": "Trazodone",
    "CID000003158": "Triamcinolone",
    "CID000003161": "Triamterene",
    "CID000003168": "Trimethoprim",
    "CID000003198": "Valproic acid (Depakote)",
    "CID000003203": "Vancomycin",
    "CID000003222": "Verapamil",
    "CID000003249": "Warfarin (Coumadin)",
    "CID000003255": "Zidovudine (AZT)",
    "CID000003261": "Zolpidem (Ambien)"
}

class DrugNameMapper:
    def __init__(self):
        self.cid2name = COMMON_DRUG_NAMES.copy()
        self.name2cid = {v.lower(): k for k, v in self.cid2name.items()}

    def get_name(self, cid):
        return self.cid2name.get(cid, cid)

    def resolve_to_cid(self, query, available_cids):
        """Resolves user query (drug name or CID) to exact CID."""
        query_str = str(query).strip()
        query_lower = query_str.lower()

        # 1. Exact match in CID
        if query_str in available_cids:
            return query_str

        # 2. Exact match in Name
        if query_lower in self.name2cid:
            cid = self.name2cid[query_lower]
            if cid in available_cids:
                return cid

        # 3. Partial match in Name
        for name_lower, cid in self.name2cid.items():
            if query_lower in name_lower and cid in available_cids:
                return cid

        # 4. Partial match in CID
        for cid in available_cids:
            if query_lower in cid.lower():
                return cid

        return query_str
