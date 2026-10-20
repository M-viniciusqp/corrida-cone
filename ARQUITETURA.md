# Arquitetura do corrida-cone

```
 Main.py  (so entrada/saida)                 sim/closedloop.py (simulador)
   le 2 cameras, manda (speed, turn)           renderiza 2 cameras + fisica do carro
        \                                        /
         +------------>  piloto.py  <-----------+      <- o MESMO codigo roda nos dois
                         Piloto.step(imagens) -> (turn, speed)
                           |
     +---------------------+-------------------------------+
     | vision.py           | cones.py                      | manobra.py
     | LaneVision          | ConeVision  (acha o cone)     | Manobra: SEGUIR -> FREAR ->
     |  erro, curvatura,   | ConeAvoider (posicao do cone, |   ANALISAR -> DESVIAR
     |  linhas, lado da    |   lado, deslocamento)         |   (classifica a faixa do cone
     |  continua           |                               |    com o carro PARADO)
     +---------------------+-------------------------------+
                           | controller.py (PID + velocidade)
   registro.py: grava logs/<data>/ (eventos.txt, telemetria.csv, fotos/ de cada decisao)
   config.py:   TODOS os numeros ajustaveis
```

## Fluxo de um cone
1. **SEGUIR**: anda seguindo a faixa; o `ConeAvoider` so observa o cone.
2. **FREAR**: cone na altura `CONE_GATILHO_ALTURA` -> motores a zero; espera o carro parar de
   verdade (cone estavel na imagem + chao da camera da pista quieto).
3. **ANALISAR**: carro parado, mede o cone de novo e **classifica a faixa** (votos: pontilhada
   na imagem do cone, camera da pista contra as linhas, regra do lado). Cone de outra faixa ->
   ignora; na faixa do carro -> escolhe o lado.
4. **DESVIAR**: decisao congelada (lado + posicao), direcao nunca vira pro lado do cone; termina
   quando o desvio acaba e o carro volta pra faixa.

## Como testar sem o carro
```
python sim/cenarios.py --pista S --todas          # uma bateria
python sim/comparar_variantes.py final ovalalt 8  # compara configs (CHAVE=valor no 4o arg)
```
O simulador usa velocidades maiores (SIM_VEL55=1) e e otimista: serve pra achar defeitos,
nao pra afinar numeros.

## Depois de uma corrida
Abra `logs/<data>/`: `eventos.txt` (o que decidiu e por que), `telemetria.csv` e `fotos/` (uma
imagem pista|cone a cada decisao). Mande essa pasta pra analise em vez de prints do terminal.
