# notebooks/

A single thin notebook used as the **Google Colab entry point**. It contains no project
logic — it only:

1. clones this repo,
2. installs `requirements.txt`,
3. mounts Google Drive (to cache the dataset and results),
4. runs code from `src/` on the GPU.

Open it in Colab via **File → Open notebook → GitHub**. Kept deliberately minimal because
`.ipynb` files version poorly (they embed outputs as JSON).
