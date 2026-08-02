# ConfigProfiler — Journal d'exécution technique

**Statut** : document vivant, complémentaire au document de référence. Le document de référence trace les *décisions de conception* ; celui-ci trace les *frictions d'exécution réelles, pièges rencontrés, et points à ne jamais oublier*. Mis à jour à chaque session de travail, jamais réécrit après coup.

**Usage** : avant de reprendre le travail chaque jour, relire la section "Points de vigilance actifs" en premier — c'est la liste courte de ce qui peut mordre si on l'oublie.

---

## Jour 1 — 2 août 2026

### Environnement — décisions et corrections

- **Windows natif écarté pour ExecuTorch, WSL2/Ubuntu retenu à la place.** Le support Windows natif d'ExecuTorch est officiellement expérimental ; la quasi-totalité du tooling (scripts, exemples, recette Qwen3) suppose un environnement Unix. Décision prise après vérification, pas après un échec de build.
- **Python 3.14 (version par défaut de l'Ubuntu WSL installée) incompatible.** Les wheels ExecuTorch ciblent 3.12/3.13. Python 3.12 installé séparément via le PPA `deadsnakes`, venv dédié créé avec `python3.12 -m venv venv`.
- **Ne jamais travailler depuis `/mnt/c/...` dans WSL** — c'est le disque Windows monté, lent et source de problèmes de permissions pour un build C++. Tout le travail se fait dans `~` (home Linux natif, `/home/mohammad_aima`).
- **`build-essential` et `cmake` ne sont pas installés par défaut sur une Ubuntu WSL fraîche.** Nécessaires pour compiler `pytorch_tokenizers` (extension pybind11) et le reste du build ExecuTorch. Installés via `sudo apt install build-essential cmake -y`.
- **CoreML (backend Apple) est activé par défaut dans `install_executorch.sh`, même sur Linux, où il ne peut pas compiler.** Provoque un échec de build (`coreml_inmemoryfs` / `executorchcoreml`) sans rapport avec le projet. **Correction obligatoire à chaque nouvel environnement** :
  ```bash
  CMAKE_ARGS="-DEXECUTORCH_BUILD_COREML=OFF" ./install_executorch.sh
  ```
- **`huggingface-cli` est déprécié**, remplacé par `hf`. Utiliser `hf auth login`, `hf download`, etc.
- **Le copier-coller de token dans un prompt interactif (`hf auth login` sans argument) peut échouer silencieusement** (header `Bearer` vide → erreur `LocalProtocolError`). Contournement fiable : `export HF_TOKEN=...` puis `hf auth login --token $HF_TOKEN`.
- **adb reste côté Windows/PowerShell, pas WSL** — décision assumée pour éviter la complexité de `usbipd-win` (partage USB vers WSL). Seuls l'export et l'inférence tournent en WSL ; le push/run sur device se fera depuis PowerShell.
- **Docker Desktop consomme de la RAM/CPU WSL2 en arrière-plan** — à fermer (`wsl --shutdown` si besoin de libérer vraiment la mémoire) avant les sessions lourdes (build, export, collecte).

### Écarts entre le README officiel et la réalité du CLI installé

- **Le README `examples/models/qwen3/README.md` documente des flags obsolètes pour `native.py`** : `--pte` et `--tokenizer` n'existent plus. Flags réels (vérifiés via `--help`) : `-f` (pte), `-t` (tokenizer). **Toujours vérifier `--help` sur la version installée avant de faire confiance à un README** — le repo ExecuTorch évolue vite.
- Chemin `4b_config.json` dans le README contient une incohérence (`examples/models/config/qwen3/` au lieu de `examples/models/qwen3/config/`) — non bloquant pour nous (on utilise 0.6B), mais à corriger si un jour on exporte la variante 4B.

### Résultat obtenu — export Qwen3-0.6B

- Export réussi : `qwen3_0_6b.pte`, **468 633 472 octets (~468 Mo)**, config `qwen3_xnnpack_q8da4w.yaml` (XNNPACK, quantization 8da4w).
- Test d'inférence réussi sur CPU WSL (pas encore Arm mobile réel) : génération cohérente en anglais, `Prefill time: 1.96s`, **`Generation: 9.57 tok/s`**.
  - ⚠️ **Ce chiffre de tok/s n'est PAS une mesure de référence pour le dataset énergie/thermal** — c'est du CPU x86 sous WSL2, pas le Tensor G2 du Pixel 7a. À ne jamais citer dans le write-up comme une performance du device cible.
- Texte de sortie répétitif ("What is the president..." en boucle) — **attendu et non-bloquant** : modèle 0.6B très quantizé, aucune pénalité de répétition (`repetition_penalty`) activée dans ce test minimal. Ne remet pas en cause la validité de l'export.

### ⚠️ Points de vigilance actifs — à ne pas oublier pour la suite

- **`Warning - given vocab_size in params is unequal to tokenizer vocab size.`** — apparu au premier run, n'a pas empêché une génération cohérente cette fois. **À surveiller activement pendant la collecte du dataset de fidélité jargon (section 5, étape 1 du document de référence)** : si des tokens rares (jargon technique protégé) se comportent bizarrement ou disparaissent, revérifier ce warning en premier avant de conclure à un effet de quantization — pourrait être une confusion tokenizer/modèle plutôt qu'un vrai résultat.
- **Toute commande multi-lignes avec `\` de retour à la ligne s'est révélée source d'erreurs de frappe répétées** (option mal coupée, espace en trop). Préférer les commandes sur une seule ligne continue pour tout ce qui est copié-collé dans le terminal, en particulier pour les commandes d'export/run réutilisées plusieurs fois (matrice de collecte à venir).
- **Le runner C++ natif n'est pas encore buildé** — nécessaire avant tout test sur le Pixel 7a physique (prochaine étape).
- **`collect_benchmarks.py` n'a pas encore été adapté à Qwen3** — reste un script générique. Ne pas commencer avant que le run on-device réel soit confirmé, pour éviter d'adapter un script sur une base non validée.

---

## Modèle à réutiliser pour les prochaines entrées

```
## Jour N — date

### Ce qui a été fait
### Écarts entre la doc/le plan et la réalité rencontrée
### Résultat obtenu (chiffres, fichiers produits)
### ⚠️ Points de vigilance actifs — à ne pas oublier pour la suite
```
