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

### Runner C++ natif — build et test

- **`cmake --preset llm` seul ne suffit pas** à produire `llama_main` — il ne fait que configurer/builder les libs de base (`executor_runner` générique). Le vrai runner LLM nécessite la procédure séparée du README `examples/models/llama/README.md`, Step 3 :
  ```bash
  cd examples/models/llama
  cmake --workflow --preset llama-release
  ```
  Résultat : `cmake-out/examples/models/llama/llama_main` (~qq secondes de build, les libs de base étaient déjà compilées par l'étape précédente).
- **Le runner C++ (`llama_main`) attend un prompt avec chat template appliqué manuellement**, contrairement au runner Python — format Qwen3 : `<|im_start|>user ... <|im_end|><|im_start|>assistant`.
- **Warning RE2 anticipé par le README, résolu automatiquement** : `Re2 failed to compile regex ... invalid perl operator: (?!` → fallback PCRE2 déclenché tout seul (`Creating PCRE2 regex`), génération réussie sans avoir besoin de recompiler avec `-DSUPPORT_REGEX_LOOKAHEAD=ON`. Ne pas s'inquiéter si ce warning réapparaît.
- **Métriques du test C++ (WSL x86 CPU, toujours pas représentatives du Pixel 7a)** : prefill 59.09 tok/s, decode 15.16 tok/s.

### ⚠️ Nouveau point de vigilance — mode "thinking" de Qwen3 activé par défaut

Le premier test avec chat template a produit une sortie commençant par `<think>` (dupliqué deux fois), un raisonnement interne avant la réponse finale — comportement par défaut de Qwen3, pas un bug.

**Impact potentiel sur le projet, à trancher avant la collecte du dataset de fidélité (section 5, étape 1)** :
- Le mode thinking ajoute des tokens et du bruit non pertinents pour une tâche de traduction — risque de fausser la mesure de préservation du jargon si le raisonnement interne contient ou déforme les termes protégés.
- À vérifier : le chat template Qwen3 permet généralement de désactiver ce mode (souvent via un flag type `enable_thinking=False` dans le template, ou une balise différente). À investiguer avant la collecte du dataset FR→EN, pas pendant.
- Décision à dater explicitement une fois tranchée, cohérent avec la méthode de rigueur du document de référence.

### Débogage sans fil — connexion établie

- Débogage sans fil configuré et connecté avec succès dès le jour 1 (`adb pair` puis `adb connect`), en avance sur le besoin réel (n'était nécessaire qu'à partir de la collecte de données, section 3 du README-collecte) — anticipé pour éviter d'avoir à ressortir le câble USB plus tard.
- Identifiant device affiché sous forme mDNS verbeuse (`adb-35311FDH200594-MFPiie._adb-tls-connect._tcp`) plutôt qu'un simple `IP:PORT` — comportement normal de la découverte automatique Android récente, pas une erreur. Utiliser cet identifiant complet (ou le retrouver via `adb devices`) pour les commandes `adb -s ...` à venir.
- **Point de vigilance déjà identifié dans le document de référence, à ne pas oublier au moment de la collecte** : vérifier `status: discharging` via `adb shell dumpsys battery` avant chaque session (le Wi-Fi seul ne garantit pas l'absence de charge si le câble reste branché par erreur).

### Déploiement sur Pixel 7a — fichiers en place, runner ARM64 restant à faire

- **Modèle et tokenizer poussés avec succès sur le device** dans `/data/local/tmp/configprofiler/` (`qwen3_0_6b.pte` 468 Mo, `tokenizer.json` 11 Mo), via USB (27-30 MB/s, quelques secondes).
- **`adb.exe` (process Windows) ne comprend pas les chemins absolus WSL/Linux** (`/home/mohammad_aima/...`) même résolus par `$(...)` en bash — l'argument est passé tel quel à un binaire Windows qui ne sait pas l'interpréter. Solution : toujours copier le fichier source dans le dossier courant avant `adb.exe push`, utiliser un chemin relatif simple.
- **Débogage Wi-Fi trop lent pour un transfert de fichier volumineux** (468 Mo à 1% après plusieurs minutes, projection >30 min) — basculé sur USB pour ce push précis (16s, 27.8 MB/s). Wi-Fi reste la bonne méthode pour les *mesures* de batterie/watts (évite le biais de charge), mais pas pratique pour déployer de gros fichiers. Pattern à retenir : USB pour déployer, Wi-Fi pour mesurer — **toujours redébrancher le câble avant toute collecte réelle**.
- **⚠️ ÉTAPE RESTANTE, bloquante pour un test on-device réel** : le `llama_main` buildé aujourd'hui (`cmake --workflow --preset llama-release`) est compilé en **x86_64** (architecture WSL/PC), pas en **ARM64** (Pixel 7a / Tensor G2). Il ne fonctionnera pas tel quel sur le device. Nécessite une **cross-compilation via le NDK Android** — non commencée, à faire en premier le jour 2, avant toute autre tâche.

## Jour 2 — 3 août 2026

### Cross-compilation Android — NDK et build

- **NDK r28c installé manuellement** (pas via Android Studio) : téléchargé depuis `https://dl.google.com/android/repository/android-ndk-r28c-linux.zip` (~720 Mo), vérifié par SHA1 (`a7b54a5de87fecd125a17d54f73c446199e72a64`), décompressé dans `~/android-ndk-r28c`. Variable `ANDROID_NDK` exportée et ajoutée à `~/.bashrc` pour persister entre sessions.
- **Aucun preset combiné "Android + LLM runner" n'existe** dans ExecuTorch — le preset `android-arm64-v8a` (`tools/cmake/preset/android.cmake`) est en réalité complet (contient bien `EXECUTORCH_BUILD_EXTENSION_LLM_RUNNER ON`), mais une première tentative avec des flags composés à la main (`-DANDROID_ABI=...` + flags XNNPACK seuls) a échoué avec `ExecuTorch must be installed with EXECUTORCH_BUILD_EXTENSION_LLM_RUNNER enabled`. **Leçon : toujours utiliser un preset existant (`cmake --list-presets`) plutôt que de recomposer des flags à la main** — le preset `android-arm64-v8a` seul, combiné avec `-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_PLATFORM=android-23`, suffit.
- **`CMAKE_BUILD_TYPE` n'est PAS fixé par défaut par le preset `common`** — un premier essai a silencieusement produit un build **Debug** (non optimisé). ⚠️ **Toujours vérifier `CMAKE_BUILD_TYPE : Release` dans le résumé de config avant de lancer le build** — un binaire Debug utilisé par erreur pour la collecte de données (Jour 3+) fausserait complètement les mesures de tokens/sec.
- **`CMAKE_INSTALL_PREFIX` non fixé également** → un premier `cmake --build ... --target install` a tenté d'écrire dans `/usr/local/lib` (système, `Permission denied`). Toujours passer `-DCMAKE_INSTALL_PREFIX=<dossier local>` explicitement pour un build Android, comme pour le build x86 d'hier.
- **⚠️ Piège rencontré, cause probablement un guillemet/échappement mal placé dans une commande de reconfiguration intermédiaire** : l'installation a atterri dans un dossier nommé littéralement `cmake-out-android-arm64-v8a*` (astérisque inclus dans le nom, pas interprété comme wildcard). Contenu confirmé identique/complet (structure `bin/include/lib/share`), juste mal nommé — renommé proprement en `cmake-out-android-install` via `mv "cmake-out-android-arm64-v8a*" cmake-out-android-install`. **Si ça se reproduit : toujours vérifier `ls` après une installation avant de supposer que le chemin attendu est le bon** — ne pas faire confiance aveuglément au nom de dossier passé en paramètre sans vérifier ce qui a été réellement créé.
- **Séparer `cmake --build` et `cmake --install` en deux commandes distinctes a cassé le cache d'un sous-projet externe (`flatcc`, géré via ExternalProject)** : la deuxième commande seule a échoué avec `file INSTALL cannot find .../libflatccrt.a`, alors que le fichier existait au moment du premier échec (permission denied). **Leçon : toujours relancer `cmake --build ... --target install` en une seule commande complète plutôt que de scinder build et install séparément** — au moins pour ce repo, qui a des dépendances externes fragiles à la réinvocation partielle.

### Chemins de build à retenir pour la suite

- `cmake-out-android-arm64-v8a/` = dossier de build brut (objets compilés avant install) — ne pas l'utiliser comme référence pour les libs finales.
- `cmake-out-android-install/` = dossier d'installation propre (structure `bin/include/lib/share`) — **c'est celui-ci qui sert de référence pour builder le runner llama Android et pour tout ce qui suit**.
- `cmake-out-android/` (sans le suffixe ABI) = résidu d'une tentative de configuration antérieure avec flags composés à la main (celle qui a échoué sur `LLM_RUNNER`) — à ignorer, pas supprimé par précaution mais non utilisé.

### 🎯 Premier test réel on-device — Pixel 7a, Tensor G2

**Pipeline complet validé de bout en bout** : export → cross-compilation ARM64 → déploiement → inférence, sur le device cible réel.

Commande utilisée :
```bash
adb.exe -s <serial> shell "cd /data/local/tmp/configprofiler && ./llama_main --model_path=qwen3_0_6b.pte --tokenizer_path=tokenizer.json --prompt=..."
```

**Résultats — premières mesures réellement représentatives du Tensor G2** (contrairement aux chiffres WSL du Jour 1, explicitement non représentatifs) :
- Prefill : **73.45 tok/s**
- Decode : **22.00 tok/s**
- 114 tokens générés, 13 tokens de prompt

⚠️ **Contexte de cette mesure, à ne pas oublier** : test effectué **câble USB branché** (nécessaire pour le push du binaire). Le tok/s n'est probablement pas biaisé par la charge, mais **ne pas réutiliser ce chiffre comme référence "watts" ou "conditions batterie contrôlées"** — pour ça, refaire un test en conditions Wi-Fi/`discharging` confirmé, conformément à la méthodologie du README-collecte.

**Comportement cohérent avec WSL** : même warning RE2 → fallback PCRE2 automatique (pas de `-DSUPPORT_REGEX_LOOKAHEAD=ON` nécessaire), même mode "thinking" Qwen3 actif par défaut — confirme que le comportement du modèle est stable entre environnements, bon signe de fiabilité pour la suite.

 — apparu au premier run (Jour 1), n'a pas empêché une génération cohérente. **À surveiller activement pendant la collecte du dataset de fidélité jargon** (section 5, étape 1 du document de référence) : si des tokens rares se comportent bizarrement, revérifier ce warning avant de conclure à un effet de quantization.
- **Mode "thinking" de Qwen3 activé par défaut** (Jour 1) — génère des balises `<think>` avant la réponse finale. À vérifier/désactiver via le chat template avant la collecte du dataset de fidélité jargon, pour éviter de fausser la mesure avec du texte de raisonnement interne non pertinent.
- **Toute commande multi-lignes avec `\` de retour à la ligne s'est révélée source d'erreurs de frappe répétées.** Préférer les commandes sur une seule ligne continue, en particulier pour les commandes d'export/run/build réutilisées plusieurs fois (matrice de collecte à venir).
- **Le runner ARM64 n'est pas encore buildé** (`cmake-out-android-install` contient les libs ExecuTorch, mais pas encore `llama_main` pour Android — étape suivante immédiate).
- **`collect_benchmarks.py` n'a pas encore été adapté à Qwen3.** Ne pas commencer avant qu'un run réel on-device (Pixel 7a) soit confirmé fonctionnel.
- **Toujours vérifier `CMAKE_BUILD_TYPE` et `CMAKE_INSTALL_PREFIX` explicitement dans tout nouveau build CMake de ce repo** — aucun des deux n'a de défaut sûr, contrairement à ce qu'on pourrait supposer.

---

## Modèle à réutiliser pour les prochaines entrées

```
## Jour N — date

### Ce qui a été fait
### Écarts entre la doc/le plan et la réalité rencontrée
### Résultat obtenu (chiffres, fichiers produits)
### ⚠️ Points de vigilance actifs — à ne pas oublier pour la suite
```
