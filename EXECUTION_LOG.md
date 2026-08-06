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
### Deuxième test on-device — conditions propres (Wi-Fi, discharging confirmé)

Même prompt, même modèle, câble débranché, `status: discharging` reconfirmé avant le run.

**Résultats** :
- Prefill : **103.17 tok/s**, Decode : **36.81 tok/s**

**⚠️ Écart notable vs le test d'hier sous USB (73.45 / 22.00 tok/s) — à ne pas surinterpréter.** Un seul run par condition (n=1 vs n=1) : impossible de distinguer à ce stade un effet réel (throttling thermique lié à la charge, changement de gouverneur CPU) d'une simple variance run-à-run ou d'un effet d'ordre (cache système déjà chaud au 2e run). **Ne pas citer ces deux chiffres comme preuve d'un effet USB dans le write-up** tant que la matrice de collecte n'aura pas confirmé ça avec plusieurs runs répétés par condition — exactement le genre de variable que `collect_benchmarks.py` est censé contrôler méthodiquement. À garder comme signal à vérifier, pas comme résultat.

### collect_benchmarks.py adapté à Qwen3 — script réécrit, testé en dry-run

**Réutilisé sans modification** : classe `Device` entière (batterie, thermal, freq/throttling) — validée sur Pixel 7a, aucune raison de retoucher.

**Réécrit** : `run_on_device_benchmark()` appelle réellement `llama_main` avec le chat template Qwen3, parse le JSON `PyTorchObserver` réel (confirmé sur les deux tests on-device) au lieu de l'ancien regex générique qui ne correspondait à aucune sortie réelle du runner.

**Testé en `--dry-run --probe-device` et `--dry-run` complet** (avec `force` sur les paliers batterie simulés) : pipeline de bout en bout validé — paliers, warmup, reprise, écriture CSV, parsing JSON tous fonctionnels avant tout run réel sur device.

**Flag threads confirmé via `llama_main --help` sur device** : c'est `-cpu_threads` (int32, défaut -1 = heuristique auto), pas `--threads`. `gflags` accepte les deux styles de tirets (`-flag` et `--flag`), donc `--cpu_threads=N` fonctionne — cohérent avec `--tokenizer_path` qui avait déjà fonctionné en style double-tiret dans les tests précédents.

**⚠️ Deux TODO restants avant une vraie collecte complète** (documentés en tête du script) :
1. Un seul `.pte` exporté à ce jour (`q8da4w`) — la matrice int8/int4 réelle nécessite d'exporter et pousser un `.pte` par niveau de quantization.
2. Mode "thinking" Qwen3 non tranché — décider de le désactiver via le chat template ou de l'assumer comme partie du protocole, avant de lancer une collecte qui servira de référence.

### Wrapper `adb` créé pour les appels Python `subprocess`

**Piège** : `adb.exe` fonctionne en ligne de commande directe dans WSL (interop Windows native), mais `subprocess.run(["adb", ...])` en Python échoue avec `FileNotFoundError: No such file or directory: 'adb'` — les alias bash ne s'appliquent pas aux appels subprocess non-shell, il faut un vrai exécutable nommé `adb` dans le PATH.

**Solution appliquée** — wrapper minimal :
```bash
mkdir -p ~/.local/bin
cat > ~/.local/bin/adb << 'EOF'
#!/bin/bash
exec adb.exe "$@"
EOF
chmod +x ~/.local/bin/adb
export PATH="$HOME/.local/bin:$PATH"          # session courante
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc   # permanent
```
Testé et fonctionnel : `adb devices` (commande directe) et tout script Python utilisant `subprocess.run(["adb", ...])` fonctionnent désormais de façon identique.

**Piège annexe rencontré en cours de route** : un fichier `collect_benchmarks.py` préexistant, appartenant à `root`, bloquait la copie du nouveau script (`Permission denied`) — résolu par `sudo rm` avant de recopier. Origine du fichier root non identifiée (antérieur au démarrage connu du projet), à surveiller si d'autres fichiers `root`-owned apparaissent dans `~/executorch`.

### 🎯 Premier run réel complet via collect_benchmarks.py — pipeline end-to-end validé

Commande : `python3 collect_benchmarks.py --serial 192.168.1.63:40889 --reps 1 --quantizations q8da4w --threads 2` (paliers forcés avec `force`, test de validation du pipeline, pas une vraie collecte).

**Résultat** : CSV correctement rempli, toutes colonnes présentes (prefill/decode tok/s, prompt/generated tokens, model_load_ms, watts, throttled, freq_drop). Le flag `--cpu_threads=2` a bien été appliqué sans erreur.

**⚠️ Deux questions méthodologiques ouvertes, à trancher avant toute vraie collecte de référence** :

1. **`throttled=True` sur 100% des runs de ce test** (freq_drop_pct 23-37%). Probablement un artefact de runs enchaînés sans vraie récupération thermique (paliers forcés, pas de vraie attente) — à revérifier avec de vrais paliers respectés. Si ça persiste avec un vrai protocole (paliers réels, temps de repos entre blocs), ce serait suspect et mériterait une investigation plus profonde.

2. **`watts_estimated` anormalement bas pendant l'inférence active (0.30–1.11 W) vs le probe à l'arrêt plus tôt dans la journée (3.53 W).** Contre-intuitif — l'inférence CPU active devrait consommer plus, pas moins, qu'un état de repos. Hypothèse principale : `battery_watts()` est appelé dans `run_single()` **après** la fin de l'inférence (`bench = run_on_device_benchmark(...)` puis `watts = device.battery_watts()`), donc mesure un instant post-inférence où le CPU a déjà pu redescendre, pas la consommation pendant le run. **À corriger avant la vraie collecte** : envisager un échantillonnage pendant l'inférence (ex. thread de sampling en parallèle, ou au minimum immédiatement avant ET après pour encadrer), plutôt qu'un seul point après coup. Sujet à trancher explicitement, pas à laisser filer — c'est le genre de faille qu'un jury technique pourrait repérer en Q&A ("comment mesurez-vous la puissance PENDANT l'inférence ?").

3. **Observation non expliquée, à vérifier avec plus de données** : decode ~48-49 tok/s sur ce run enchaîné, contre 22.00 et 36.81 tok/s sur les deux tests isolés d'hier (mêmes prompt/modèle, threads par défaut vs. `--cpu_threads=2` explicite ici). Piste probable : cache OS/page cache déjà chaud sur des runs qui s'enchaînent vs. cold start à chaque test isolé — mais pourrait aussi être un effet réel du flag `cpu_threads`. Ne pas conclure sans réplication (plusieurs valeurs de threads, plusieurs reps).

### Correction méthodologique — mesure des watts par échantillonnage pendant l'inférence

**Défaut corrigé** : `battery_watts()` était appelé une seule fois, après la fin de l'inférence — mesurait un état déjà redescendu, pas la consommation réelle pendant le run (cause probable des watts anormalement bas observés au run précédent : 0.30–1.11 W actif vs 3.53 W au repos).

**Solution implémentée** : classe `WattsSampler`, thread dédié qui appelle `battery_watts()` toutes les 0.4s pendant toute la durée de l'appel bloquant à `llama_main` (encadre précisément l'inférence, pas juste avant/après). Le CSV rapporte maintenant `watts_mean`, `watts_min`, `watts_max`, `watts_n_samples` au lieu d'une seule valeur post-hoc — transparence totale sur la méthode, exploitable directement pour répondre à une question de jury sur "comment mesurez-vous la puissance pendant l'inférence".

**Testé en dry-run** : schéma CSV correct, `watts_n_samples=1` en dry-run (attendu — l'appel simulé est instantané, pas assez de temps pour plusieurs échantillons à intervalle 0.4s). Sur device réel (inférence de 4-8s observée), attendre 10-20 échantillons par run — **à reconfirmer avec un vrai run device avant de considérer cette correction validée**.

### Approfondissements méthodologiques Jour 2 — 4 ajouts, testés en dry-run

En réponse à "comment aller plus loin pour un jury exigeant", 4 corrections/ajouts à coût faible et impact crédibilité élevé, tous implémentés et testés :

1. **Baseline idle** (`--idle-baseline-s`, défaut 2s) : mesure la consommation au repos (écran éteint, aucune inférence) juste avant chaque run. `watts_delta_mean = watts_mean - baseline_watts_mean` donne le coût attribuable à l'inférence, pas une valeur brute sans référence.
2. **Contrôle de l'état écran** (`Device.ensure_screen_off`) : variable confondante non maîtrisée jusqu'ici — un petit modèle quantizé peut consommer moins que l'écran allumé. Forcé éteint avant chaque baseline (idempotent, ne rallume jamais par erreur).
3. **Dimension `cache_state` (cold/warm)** via `--inter-rep-pause-s` : l'écart de tok/s entre runs isolés (22-37 tok/s, Jour 1-2) et runs enchaînés (~48 tok/s, run de validation) n'est plus une anomalie non expliquée — c'est désormais une variable expérimentale explicite et contrôlable, pas un angle mort.
4. **`--check-throttle-access`** : diagnostic à lancer avant toute vraie collecte, confirme si `scaling_cur_freq`/`cpuinfo_max_freq` sont lisibles (pas bloqués par SELinux comme `thermal_zone`) — évite de présenter `throttled` comme plus fiable qu'il ne l'est.

**Testé en dry-run** : `--check-throttle-access` fonctionne, pipeline complet avec `--idle-baseline-s 0.5 --inter-rep-pause-s 0.5` produit bien `cache_state=cold` et `watts_delta_mean` calculé ; comportement par défaut (`--idle-baseline-s 0`) confirme la rétrocompatibilité avec note explicite `baseline désactivée`.

**⚠️ À faire avant la vraie collecte, sur device réel (pas juste dry-run)** :
- Lancer `python3 collect_benchmarks.py --serial <IP:PORT> --check-throttle-access` pour confirmer si la lecture cpufreq est vraiment accessible sur le Pixel 7a (jamais testé en conditions réelles, seulement en mock).
- Reconfirmer que `ensure_screen_off()` fonctionne réellement sur le device (le keyevent 26 pourrait avoir un comportement différent selon l'état de verrouillage de l'écran — à vérifier).
- Décider de la valeur par défaut de `--inter-rep-pause-s` pour la vraie collecte (0 = reproduit le comportement du run de validation, >0 = protocole plus rigoureux mais collecte plus longue).

**✅ Confirmé sur device réel (Jour 2)** : `--check-throttle-access` montre que la lecture `scaling_cur_freq`/`cpuinfo_max_freq` fonctionne sans blocage SELinux sur le Pixel 7a (contrairement à `thermal_zone`) — 8 cœurs détectés, fréquences lisibles. Le `throttled` du dataset sera donc basé sur la vraie fréquence CPU, pas un repli approximatif. Un des deux points de vigilance de la liste ci-dessous est donc levé.

### 🎯 Pipeline de collecte validé de bout en bout, dataset propre obtenu

Après correction d'un souci de fichier CSV à schéma mixte (résultat d'un ancien fichier non renommé avant un nouveau run — toujours vérifier `results/` avant un nouveau test si le schéma du script a changé), premier dataset propre obtenu :
- 12 runs (1 config × 3 paliers batterie × 2 conditions thermiques × 2 [warmup+rep])
- Schéma CSV complet et cohérent (header et données alignés)
- `watts_delta_mean` désormais cohérent et positif (3.4-5.0 W au-dessus du baseline idle), résolvant l'anomalie détectée plus tôt dans la journée

**Bilan Jour 2** : pipeline de collecte scientifiquement défendable, pas juste fonctionnel — export, cross-compilation ARM64, déploiement, mesure watts corrigée pendant l'inférence, baseline idle, contrôle écran, dimension cache_state, diagnostic throttle, tous validés sur device réel.

### Approfondissement — séries temporelles watts + durées de phase réelles (au-delà du plan Jour 2)

En creusant la variance watts_min/watts_max observée, deux ajouts distincts, avec une limite honnêtement assumée plutôt qu'une fausse précision :

1. **`prefill_duration_ms` / `decode_duration_ms`** : calculées uniquement à partir de timestamps **device** (`inference_start_ms`, `prompt_eval_end_ms`, `inference_end_ms` du JSON PyTorchObserver) — même horloge des deux côtés du calcul, donc fiable et précis.
2. **`watts_samples_json`** : série brute `[(elapsed_s, watts), ...]` par run, horodatée en secondes relatives depuis le début de l'échantillonnage (horloge **hôte**/WSL).

**⚠️ Limite explicitement NON résolue aujourd'hui, à ne pas contourner par une fausse précision** : les deux horloges (device pour les durées de phase, hôte pour les échantillons watts) ne sont **pas synchronisées**. Corréler un échantillon watts précis à "on est en train de faire du prefill" ou "du decode" nécessiterait soit une synchronisation d'horloge device/hôte, soit un marqueur explicite émis par le runner au moment de la transition prefill→decode. **Ne pas produire de graphique ou d'affirmation "watts pendant le prefill = X" tant que ce point n'est pas résolu** — le mensonge par excès de précision serait pire que l'absence de résultat sur ce point. Piste pour plus tard si le temps le permet : ajouter un `time.sleep()` calibré ou un ping ADB au tout début de l'inférence pour établir un point d'ancrage commun entre les deux horloges.

**Testé en dry-run**, schéma CSV cohérent (`prefill_duration_ms`, `decode_duration_ms`, `watts_samples_json` bien remplis). **Pas encore testé sur device réel avec cette version** — à faire avant la prochaine vraie collecte.

### Résultats réels — durées de phase et série watts, premier vrai signal exploitable

Run réel confirmé : `prefill_duration_ms=178`, `decode_duration_ms=2333` — cohérent (prefill court sur 13 tokens, decode domine ~93% du temps total).

**Série watts observée** (`watts_samples_json`) : `[0.498s→0.49W], [1.861s→5.00W], [2.968s→9.30W], [3.981s→8.68W]` — vraie rampe de montée en charge, pas du bruit. Lecture qualitative plausible : le premier échantillon bas coïncide probablement avec le chargement du modèle (observé 850-2500ms sur runs précédents), puis la puissance grimpe une fois l'inférence CPU réellement engagée. **Reste une lecture qualitative** — la limite d'horloge device/hôte documentée plus haut s'applique toujours, pas d'attribution précise de phase à ce stade.

**⚠️ Découverte méthodologique imprévue** : l'intervalle réel entre échantillons (1.0-1.4s) ne respecte pas le `interval_s=0.4` configuré. Cause identifiée : `battery_watts()` fait deux appels adb séquentiels (`current_now` puis `voltage_now`), chacun avec son propre aller-retour — le coût réel par échantillon dépasse largement le `time.sleep(0.4)` entre deux tentatives. **Résultat : sur une inférence de ~4s, seulement 4 échantillons obtenus au lieu des ~10 attendus.** Pas bloquant pour la suite (les données restent valides), mais à noter dans le write-up si `watts_n_samples` est cité comme mesure de résolution temporelle — la résolution réelle est plus grossière que le paramètre `interval_s` ne le suggère. Piste d'amélioration si le temps le permet : batcher `current_now` et `voltage_now` en une seule commande shell (`cat file1 file2`) pour réduire le nombre d'allers-retours adb par échantillon.

### Correction — échantillonnage watts optimisé (un seul appel adb au lieu de deux)

`battery_watts()` combine désormais `current_now` et `voltage_now` en une seule commande adb (`cat f1 f2`) au lieu de deux appels séquentiels. Devrait réduire le coût réel par échantillon d'environ moitié, rapprochant l'intervalle effectif du `interval_s=0.4` configuré (mesuré à 1.0-1.4s avec l'ancienne version à deux appels). **Testé en dry-run uniquement** — le vrai gain de résolution temporelle reste à confirmer sur device réel avant de le considérer acquis.

**✅ Confirmé sur device réel** : 7 échantillons obtenus (vs 4 avant l'optimisation) sur une inférence similaire — gain conforme à l'attendu (quasi doublement). Nouvelle série observée : `[0.22s→1.27W], [0.88s→2.54W], [1.58s→3.40W], [2.30s→8.22W], [2.99s→9.47W], [3.71s→8.22W], [4.38s→8.18W]` — rampe de montée en charge progressive et lisible, cohérente avec chargement/initialisation puis régime de calcul soutenu. Bon candidat de figure pour le write-up (avec la réserve d'interprétation de phase déjà documentée ci-dessus).

### Décision tranchée — mode thinking Qwen3 désactivé via soft switch `/no_think`

TODO en attente depuis hier, tranché aujourd'hui. Raisons : pollue la démo vidéo (séquence C, la plus différenciante) et fausse les métriques tokens/latence du dataset de fidélité jargon sans apporter de valeur — le projet mesure la fidélité de traduction, pas la qualité du raisonnement interne.

**Implémentation** : le hard switch (`enable_thinking=False`) est une option Python côté tokenizer HuggingFace (`apply_chat_template`), inaccessible depuis le runner C++ qui construit le prompt en texte brut. Solution retenue : soft switch textuel `/no_think` ajouté à la fin du message utilisateur dans le chat template (`build_llama_main_command`). Confirmé applicable à Qwen3 standard (pas Qwen3-VL, pas Qwen3.5, qui ont un comportement différent sur ce point — vérifié par recherche avant implémentation, pas supposé).

**Testé** : syntaxe de commande vérifiée (`/no_think` bien injecté dans le prompt templaté). **Pas encore testé sur device réel que le thinking disparaît vraiment de la sortie** — à confirmer au prochain run, avant de considérer ce point définitivement clos.

**✅ Confirmé sur device réel** : le soft switch `/no_think` fonctionne — le modèle ne génère plus de contenu de raisonnement (0 token gaspillé sur le thinking), réponse directe. **Nuance à noter** : les balises `<think>\n</think>` restent présentes dans le texte brut, mais vides — comportement documenté du soft switch (contrairement au hard switch qui les supprimerait entièrement). Si affichage du texte brut dans la démo vidéo, prévoir un nettoyage cosmétique simple (regex `<think>\s*</think>\s*` → vide) pour un rendu propre. Pas bloquant, juste à anticiper pour le tournage.

## Jour 3 — 4 août 2026

### Détection du Mode Économie d'énergie (Battery Saver) — variable cachée non contrôlée

En affinant la question des paliers batterie, découverte importante : **le seuil par défaut d'activation automatique du Battery Saver sur Pixel est 20%** — exactement la borne du palier "low" (`<20%`) du script. Si actif, ce mode bride le CPU et coupe des activités en arrière-plan — mesurer "batterie basse" sans savoir si ce mode est actif mélangerait deux effets confondus (niveau de batterie réel + throttling logiciel volontaire) dans une seule étiquette.

**Décision retenue (Option A)** : isoler l'effet batterie seul en désactivant manuellement l'activation automatique du Battery Saver avant la collecte (Paramètres → Batterie → Économiseur de batterie → Programmation et rappels), plutôt que de le documenter comme faisant partie du palier "low". Raison : le projet mesure l'effet matériel brut de la batterie, pas l'effet d'un mode logiciel — plus simple et plus clair à défendre devant un jury.

**Implémentation** : `Device.is_battery_saver_active()` (lecture `settings get global low_power`), nouvelle colonne CSV `battery_saver_active`, avertissement console immédiat si détecté actif pendant un run (pas de désactivation automatique tentée via adb — permissions non garanties, décision laissée à l'utilisateur). **Testé en dry-run uniquement** — à vérifier sur device réel avant la vraie collecte du jour, en particulier confirmer manuellement que le Battery Saver est bien désactivé avant de lancer la matrice complète.

### Dataset fidélité jargon — corpus + script de scoring créés, garde-fou méthodologique ajouté

**Livrables** : `jargon_dataset_corpus.csv` (30 phrases jargon EN utilisant le vocabulaire réel du projet — ExecuTorch, XNNPACK, KleidiAI, quantization, tokenizer, thermal throttling... + 30 phrases contrôle neutres), `score_jargon_fidelity.py` (scoring automatique par correspondance glossaire, méthode tranchée : objective + relecture manuelle en filet de sécurité pour les faux négatifs).

**Décision sur la direction de traduction** : EN→FR uniquement (pas les deux sens) — cohérent avec l'anecdote personnelle franglais, et le jargon anglais resurgissant au milieu du français reste repérable par un jury anglophone sans qu'il ait besoin de lire le français.

**Question méthodologique soulevée en cours de projet, corrigée avant toute collecte réelle** : garder le jargon intact est une pratique de traduction professionnelle légitime (pas un échec), MAIS un modèle qui échoue totalement à traduire (recopie pure de l'anglais, scénario plausible sous quantization sévère) obtiendrait un score de "préservation" parfait sans garde-fou — confondant "traduction réussie avec jargon préservé" et "aucune traduction du tout". **Ajouté** : `looks_translated` (détection de mots grammaticaux français courants), qui signale explicitement les cas `all_preserved=True` + `looks_translated=False` comme des recopies suspectes à exclure, pas de vrais succès. Testé avec un cas de recopie pure simulée — le garde-fou déclenche correctement l'alerte.

**⚠️ Reste à faire** : générer les vraies traductions sur device (adapter le prompt `llama_main` pour une tâche de traduction plutôt qu'une question factuelle) avant de pouvoir lancer un vrai scoring.

### 🐛 Bug méthodologique découvert dans le dataset de référence — throttled=100% suspect, corrigé

Le dataset de référence (36 runs, low/mid/high, vrais paliers) montrait `throttled=True` sur **100% des runs**, sans variation cohérente avec les conditions — signal suspect plutôt qu'un vrai résultat scientifique. `freq_drop_pct` variait 23-44%, avec une légère tendance low > high (36.9% vs 30.5%, possible vrai signal de fond, mais amplitude bien trop grande pour être uniquement thermique).

**Cause identifiée** : `_cpu_freqs()` lisait toutes les fréquences courantes et tous les plafonds théoriques en deux listes séparées, puis comparait `max(cur global)` à `max(max global)` — sur l'architecture big.LITTLE du Tensor G2 (LITTLE ~1.8GHz, BIG ~2.85GHz), ça revenait à comparer un cœur LITTLE actif à son propre plafond... contre le plafond du cœur BIG inactif. Résultat : une "chute" artificiellement énorme même sans surchauffe réelle, simplement parce qu'un petit modèle sur 2-4 threads ne sollicite jamais les cœurs BIG.

**Correction** : lecture appariée cœur par cœur (`cur, max` sur la même ligne, même index), sélection du cœur **le plus actif** (`cur` le plus élevé), calcul du drop sur SON PROPRE plafond uniquement. Testé en dry-run avec un mock simulant explicitement 4 cœurs LITTLE actifs + 4 cœurs BIG inactifs — confirme que le nouveau calcul sélectionne bien le cœur pertinent (`drop=5.56%`, cohérent) plutôt que le mélange entre clusters (`throttled=False` désormais en conditions normales).

**⚠️ Conséquence directe : le dataset de 36 runs collecté avant cette correction a une colonne `throttled`/`freq_drop_pct` non fiable.** Les autres colonnes (tok/s, watts, cache_state, battery_saver_active...) restent valides — seul le signal de throttling est à refaire. **Décision à prendre** : soit relancer la collecte complète avec le script corrigé (coût : refaire les 3 paliers, potentiellement plusieurs heures), soit publier le dataset actuel en excluant/annotant explicitement la colonne throttled comme non fiable dans cette version. Pas encore tranché — à décider avant de committer le dataset comme référence finale.

### Script de génération des traductions — `translate_jargon_dataset.py`, créé et testé

Réutilise `Device`/`run_on_device_benchmark` de `collect_benchmarks.py` (import local, pas de duplication) pour envoyer chaque phrase du corpus à `llama_main` avec une instruction de traduction EN→FR, `/no_think` actif, produit un CSV directement compatible avec `score_jargon_fidelity.py`.

**Piège d'extraction identifié et testé AVANT tout run réel** : la sortie brute contient l'artefact de collage déjà observé Jour 2 (`"assistantius"` — le premier token généré se colle parfois au marqueur `<|im_start|>assistant` sans espace). Extraction conçue pour couper après `</think>` plutôt qu'après le marqueur assistant, précisément pour éviter ce piège. **Testé avec le vrai exemple de sortie problématique du Jour 2** (copié tel quel depuis les logs) — extraction propre confirmée, artefact correctement écarté.

**Testé en dry-run** : logique de reprise (skip des lignes déjà traduites) fonctionnelle, CSV bien formé. Le mock de `collect_benchmarks.py` ne simule que la ligne JSON de métriques, pas de vrai texte — donc le dry-run valide la mécanique (arguments, reprise, écriture CSV) mais pas le contenu réel des traductions ; premier vrai test de contenu à faire sur device.

**⚠️ Reste à faire** : premier run réel sur device (`--limit 3` recommandé pour valider avant de lancer les 60 phrases complètes), puis lancement complet une fois confirmé propre.

## Jour 4 (soir Jour 3 / nuit) — Preuve d'optimisation Arm-spécifique (Trou 1)

### 🎯🎯🎯 Résultat majeur — comparaison fp32 naïf vs 8da4w+XNNPACK+KleidiAI sur device réel

**Contexte** : suite aux clarifications répétées du jury officiel ("prove one clear thing: what you did that makes AI better on Arm"), export et test d'une baseline fp32 naïve (aucune quantization, aucun backend Arm-spécifique) pour comparer directement à la config optimisée déjà validée.

**Étapes** :
1. Config `qwen3_portable_q8da4w.yaml` créée (quantization 8da4w SANS backend XNNPACK) — testée d'abord sur WSL : **échec runtime** (`Check failed: Tensors do not match: dtype={Float, Char, Float}`). Confirme que la représentation `IntxUnpackedToInt8Tensor` produite par la quantization 8da4w est structurellement dépendante des kernels XNNPACK — aucun kernel portable ne sait l'exécuter. Résultat gardé comme preuve à part entière ("capacité qui n'existe pas sans le backend Arm"), pas caché.
2. Config `qwen3_naive_fp32.yaml` créée (aucune quantization, aucun backend) — export réussi, `.pte` de **2 388 733 316 octets (2.39 Go)** vs **468 633 472 octets (468 Mo)** pour la version optimisée. **Ratio de taille : 5.1x**.
3. Test WSL (x86, non représentatif Arm) : 11.88 tok/s — cohérent avec l'attente que la quantization n'apporte aucun gain sans hardware Arm pour l'exploiter.
4. **Blocage stockage device** : Pixel 7a à 100% (981 Mo libres sur 110 Go) — push du `.pte` naïf (2.39 Go) impossible. Espace libéré manuellement par l'utilisateur (24 Go dispo après nettoyage) — noté comme point de storytelling honnête en soi (un modèle non quantizé ne tient même pas sur un téléphone grand public presque plein, un scénario réaliste, pas un artefact de benchmark).
5. **Test réel sur Pixel 7a (Tensor G2), fp32 naïf** :
   - `prefill_token_per_sec: 1.02762`, `decode_token_per_sec: 0.0779361`
   - 54 tokens générés en **709.1 secondes** (`aggregate_model_execution_time_ms: 709111`) — soit ~12.8s/token.

**Comparaison avec le résultat optimisé déjà mesuré (Jour 2, conditions propres Wi-Fi/discharging)** : 103.17 / 36.81 tok/s (prefill/decode).

**→ Gain mesuré : ~100x en prefill, ~472x en decode.**

**Réplication (run 2, mêmes conditions)** : `prefill_token_per_sec: 1.68701`, `decode_token_per_sec: 0.102846`. Ordre de grandeur confirmé malgré variance run-à-run normale (~50%, plausible thermique/bruit) — **moyenne sur les 2 runs : prefill 1.36 tok/s, decode 0.090 tok/s naïf, contre 103.17/36.81 optimisé → gain ~76x en prefill, ~407x en decode.** Conclusion robuste, réplication validée.

**Valeur stratégique** : répond directement et de façon spectaculaire à la question centrale du jury ("what did you do that specifically makes this better on Arm"), et correspond explicitement à un des exemples cités dans leur grille ("making something run on a constrained device that previously required more" / "unlock a use case that was not previously practical") — le modèle naïf n'est pas seulement plus lent, il est impraticable (>11 min pour une réponse courte).

### ⚠️ Points de vigilance actifs — à ne pas oublier pour la suite



















```
