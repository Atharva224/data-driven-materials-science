#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Stefan Hiemer
""" 

from matminer.featurizers.conversions import StrToComposition,CompositionToOxidComposition
from matminer.featurizers.composition import ElementProperty, OxidationStates
from matminer.featurizers.structure import DensityFeatures 

from matminer.datasets.convenience_loaders import load_elastic_tensor

def load_Steel():
    """
    Copied from
    https://nbviewer.org/github/hackingmaterials/matminer_examples/blob/main/matminer_examples/machine_learning-nb/bulk_modulus.ipynb 
    
    Jong, M. De, Chen, W., Angsten, T., Jain, A., Notestine, R., Gamst, A., 
    Sluiter, M., Ande, C. K., Zwaag, S. Van Der, Plata, J. J., Toher, C., 
    Curtarolo, S., Ceder, G., Persson, K. and Asta, M., 
    “Charting the complete elastic properties of inorganic crystalline 
    compounds”, Scientific Data volume 2, Article number: 150009 (2015)
    """
    df = load_elastic_tensor()

    unwanted_columns = ["volume", "nsites", "compliance_tensor",
                        "elastic_tensor", "elastic_tensor_original", "K_Voigt",
                        "G_Voigt", "K_Reuss", "G_Reuss"]

    df = df.drop(unwanted_columns, axis=1)

    df = StrToComposition().featurize_dataframe(df, "formula")

    ep_feat = ElementProperty.from_preset(preset_name="magpie")
    df = ep_feat.featurize_dataframe(df, col_id="composition")

    df = CompositionToOxidComposition().featurize_dataframe(df, "composition")

    os_feat = OxidationStates()
    df = os_feat.featurize_dataframe(df, "composition_oxid")

    df_feat = DensityFeatures()
    df = df_feat.featurize_dataframe(df, "structure")

    excluded = ["G_VRH", "K_VRH", "elastic_anisotropy", "formula", "material_id",
            "poisson_ratio", "structure", "composition", "composition_oxid"]
    y = df['K_VRH']
    X = df.drop(excluded, axis=1)

    # save featurized data
    X.to_csv("features-bulk.csv",index=False)
    y.to_csv("target-bulk.csv",index=False)

    return

if __name__ == "__main__":
    #create_dataset()
    load_Steel()
