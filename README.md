# ateflow

Stima di effetti causali da un DAG dichiarativo.

**La domanda a cui risponde, e solo quella:** dato un trattamento binario e un DAG,
quanto cambia l'esito al netto dei confonditori?

Tutto ciò che non serve a questa frase resta fuori, almeno per i primi tre mesi.

## Uso

Il DAG è un file di testo, una relazione per riga:

```
crowding -> led
crowding -> speed
led -> speed
led -> waiting_time
```

Dalla riga di comando:

```bash
python -m ateflow --data examples/corridor.csv --dag examples/corridor.dag \
    --treatment led --outcome speed
```

```
naive            ATE = -0.130   aggiusto per: nessuno
g-computation    ATE = +0.150  IC95% [+0.143, +0.156]   aggiusto per: crowding

bias da confondimento : -0.280
inversione di segno   : SI
refutation placebo_treatment      ok        mean=-0.0000, sd=+0.0037
refutation random_common_cause    ok        max_drift=+0.0001
```

L'ATE vero del dataset sintetico è +0.15. La stima non aggiustata ha il segno
sbagliato: `crowding` accende il LED più spesso e insieme abbassa la velocità.

Da Python:

```python
from ateflow import DAG, estimate_ate

res = estimate_ate(df, DAG.from_file("examples/corridor.dag"),
                   treatment="led", outcome="speed")
print(res.report(), res.sign_flip, res.confounding_bias)
```

## Validazione su dati reali

`examples/episodes_100_v1.csv` sono i 100 episodi HRI del dataset
[PeopleFlow](https://github.com/aasterr/PeopleFlow/tree/main/analysis): un robot
che attraversa un corridoio e decide se emettere un segnale LED (`A`), con
esito successo/timeout (`T`) e ostacoli statici come confonditore (`O`).

```bash
python -m ateflow --data examples/episodes_100_v1.csv --dag examples/hrisim.dag \
    --treatment A --outcome T --method stratification
```

ateflow riproduce al terzo decimale le stime della tesi di riferimento:
naive −0.207 (segno invertito dal confondimento), backdoor su `{O}` +0.061,
backdoor su `{Pi, O}` +0.108 sui soli 64 episodi con overlap — lo strato
`Pi=0, O=0` non contiene alcun episodio trattato e viene scartato, non riempito.
`tests/test_hrisim.py` fissa questi numeri come regressione.

## Cosa fa oggi

- DAG con controllo di aciclicità e parsing di catene (`a -> b -> c`)
- d-separazione per moralizzazione del sottografo ancestrale
- criterio di backdoor di Pearl, ricerca dell'insieme minimale e degli insiemi alternativi
- rifiuto esplicito di mediatori, collider e discendenti del trattamento
- stima per g-computation (con interazioni T*Z) e per stratificazione
- intervalli bootstrap percentile
- refutation test: placebo treatment, random common cause

## Cosa non fa (per scelta)

Trattamenti non binari, dati longitudinali, front-door, variabili strumentali,
causal discovery, effetti eterogenei per sottogruppo. Sono estensioni, non requisiti.

## Sviluppo

```bash
pip install -e ".[dev]"
python examples/make_data.py
pytest -q
```

`tests/test_estimate.py` è il test di regressione di tutto il progetto: se un
refactoring rompe il recupero dell'ATE noto, la pipeline è rotta.
