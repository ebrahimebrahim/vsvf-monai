# vsvf-monai
Experimentation with ways of integrating the fluid-based registration technique vSVF into MONAI.

# Prepare OAI dataset
Run `bash download.sh` beforehand. This wil download sample OAI data to `data/OAI`.

# Mermaid advection solver demo
Run `python mermaid_demo.py`.

This should give you sample registration performance output:
```
=====================
class 0 Dice is 0.997
class 1 Dice is 0.655
class 2 Dice is 0.630
=====================
```
