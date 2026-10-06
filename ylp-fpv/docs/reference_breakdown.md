# Análise das referências

Medido nos ficheiros, não a olho: deteção de cortes por histograma, movimento de
câmara por fluxo ótico (pan, tilt, zoom, roll e paralaxe por plano), BPM por
autocorrelação das transientes, efeitos sonoros pelo espectrograma, cor em Lab.
Ferramentas em `tools/ref_*.py`:

```bash
python3 tools/ref_sheet.py ref.mp4 sheet.jpg 1.0 256 8         # folha de contacto, 1 imagem/s
python3 tools/ref_shots.py ref.mp4 shots.json                  # cortes e movimento por plano
python3 tools/ref_audio.py ref.mp4 shots.json audio.png        # BPM, cortes no tempo, efeitos
python3 tools/ref_grade.py ref.mp4 nosso.mp4                   # impressão digital de cor
python3 tools/ref_strip.py ref.mp4 strip.jpg 38.5:39.1 110     # frame a frame de uma transição
```

## Ref 1 · Sciame, Palm Beach (16:9, 75 s)

Filme de promotor com filmagem real: drone, gimbal, ator, carro clássico.

- Montagem: 46 planos, média 1,6 s, mediana 1,2 s. As durações são múltiplos
  do tempo musical a 105,6 BPM: 1, 2, 4, 6 e 8 tempos. 80% dos cortes caem no
  tempo (menos de 60 ms de erro).
- Estrutura: fachada, logótipo sobre texturas abstratas (mármore, cerâmica),
  escultura e piscina, lifestyle (homem de fato, carro vermelho), aéreos
  zenitais, jardim e piscina em planos longos de gimbal (2 a 4,6 s), entrada na
  casa com speed ramp, interiores (sala, jantar, cozinha, quarto), detalhes de
  fachada, aéreos da cidade, água da piscina, zenital, logótipo sobre preto.
- Câmara: um movimento por plano, lento (0,02 a 0,15 da largura do quadro por
  segundo). Exceções que dão energia: a entrada na casa (38,7 s) arranca a 8
  vezes a velocidade média e trava. Avanços tipo FPV na cozinha e no quarto
  (51 a 56 s). Um contrapicado da fachada com roll de 13°/s.
- Transições: só cortes secos. A energia vem do movimento dentro do plano e do
  corte no tempo.
- Texto: só o logótipo. Nome em serifa, filete vertical, duas linhas em
  versaletes. Sobre textura no início, sobre preto no fim.
- Cor: low-key quente. Preto 4,3, branco 87,5, média 37, contraste 22,6,
  saturação 37%, b* +8,5 (L, a*, b* em Lab).
- Som: faixa contínua tipo chill house a 105,6 BPM com baixo presente. Quase
  sem efeitos: whooshes suaves em 4 cortes, acorde final sustentado sobre o
  logótipo.

## Ref 2 · Propertia (9:16, 25 s)

Reel de agência com filmagem real: gimbal e drone.

- Montagem: 35 planos. Gancho de 3,7 s com 18 cortes de 0,17 a 0,23 s (cinco
  por segundo). Depois planos de 1,0 a 1,2 s. 100% dos cortes na semicolcheia a
  166 BPM (sente-se a 83).
- Estrutura: flashes de detalhe, zenital de dia com corte seco para a mesma
  posição à noite e de volta, mergulho FPV na cozinha (1,9 larguras de quadro
  por segundo no pico, a travar), sala, jantar, escada, banheira exterior,
  quarto e passagem pela porta, baloiço, revelação da piscina por trás de
  folhagem em primeiro plano, zenital final.
- Câmara: gimbal lento na maioria. Dois planos com rotação de 12 e 18°/s. O
  mergulho FPV trava. A passagem pela porta acelera.
- Transições: só cortes secos e um match cut dia/noite.
- Texto: logótipo pequeno fixo no topo durante todo o vídeo. Mais nada.
- Cor: quente, médio. Preto 9,8, branco 84,8, média 45, contraste 19,6,
  saturação 29%, b* +10,1.
- Som: design de som forte por cima de uma faixa com vocal chops.
  - Downlifter por baixo do gancho: tom harmónico a descer de cerca de 1 kHz
    para 250 Hz em 3,7 s.
  - Impacto grave quando entra o zenital.
  - Quase silêncio no plano noturno, com um tom agudo a soar (o "ting").
  - Grande whoosh no mergulho para dentro de casa.
  - Whooshes em 11 das 13 transições depois do gancho.

## O que replicamos com fotos

| Ideia | Ref | Como fazemos |
|-------|-----|--------------|
| Cortes secos na grelha da música | 1 e 2 | 108 BPM, planos de 2 e 4 tempos, flashes de meio tempo |
| Gancho de flashes de detalhe | 2 | 8 cortes de 0,28 s, tick em cada um, downlifter por baixo |
| Logótipo sobre textura, cartão final preto | 1 | lockup sobre a madeira da ilha, cartão preto com ref. e site |
| Mergulho com speed ramp | 1 e 2 | avanço rápido que trava no corredor da cozinha, no drop |
| Planos de gimbal com paralaxe | 1 e 2 | travelling lateral, avanços lentos, pan que acelera da ilha à piscina |
| Detalhes com profundidade de campo | 1 | cortes de 2 tempos com desfoque calculado do mapa de profundidade |
| Cor low-key quente | 1 e 2 | grade ajustado por medição contra as duas referências |
| Impacto, silêncio, swell, drop | 2 | impacto e ting no logótipo, dropout, swell invertido até ao drop |

## O que não dá com fotos

- Voo contínuo de drone, zenitais e aéreos de bairro: não há fotos aéreas.
- Lifestyle (pessoas, carro) e troca dia/noite: precisa de fotos reais.
- Macro verdadeiro: com fotos de 1600 px os detalhes ficam macios.

Para a próxima: originais do fotógrafo em resolução total, fotos de drone e
zenitais, fotos ao fim do dia se existirem, e o logótipo YLP em SVG.
