# Teste Higgsfield (VM687)

Objetivo: medir se o vídeo generativo melhora o corte cinematográfico
(`config/vm687_cine_16x9.json`) sem mentir sobre a casa.

## Regra

- A IA pode mexer em luz, água, folhagem e nitidez.
- Nunca em arquitetura, materiais, vistas pelas janelas ou no que fica fora da foto.
- A câmara segue as referências: gimbal, horizonte nivelado, sem oscilação nem diagonais.
- Um clip que falhe a verificação não entra no vídeo.
- Antes de publicar, a YLP tem de aceitar movimento gerado por IA com aviso (AI Act, art. 50).

## Preparação numa sessão nova

1. Ambiente: `api.higgsfield.ai`, `docs.higgsfield.ai` e `console.higgsfield.ai` em
   Allowed domains. O SDK oficial `higgsfield-client` só chama `api.higgsfield.ai`, mais o
   anfitrião dos ficheiros gerados quando for preciso descarregá-los.
   Chave em `HF_KEY`, no formato `key-id:key-secret`, nas variáveis do ambiente ou em
   `.env.local` (ignorado pelo Git). A chave nunca vai para o chat, para commits nem para logs.
   Verificação mínima: `python3 main.py` gera um clip Seedance 2.5 de 5 s e imprime o URL.
2. Fotos: não estão no repositório. Pedir as 5 fotos e gravá-las em
   `assets/vm687/photos` com estes nomes:

   | Ficheiro | O que mostra |
   |----------|--------------|
   | `01_living.jpg` | sala inteira vista do fundo, cozinha com ilha ao fundo |
   | `02_kitchen.jpg` | cozinha, coluna de fornos em madeira à esquerda, ilha ao centro, ripado exterior ao fundo |
   | `03_kitchen_aisle.jpg` | junto à coluna de fornos, ilha à direita, sombras do ripado no chão |
   | `04_pool_view.jpg` | da ilha para a sala, janelas de correr com piscina e jardim |
   | `05_garden_room.jpg` | divisão vazia com janela de correr para o jardim e a piscina |

3. Logo: pedir o SVG, gravá-lo em `assets/brand/ylp-logo.svg` e correr
   `python3 tools/svg2png.py assets/brand/ylp-logo.svg assets/brand/ylp-logo.png`.
4. Profundidade e render de referência: `./run.sh config/vm687_cine_16x9.json`
   (descarrega o modelo de profundidade na primeira vez).
5. API: ler a documentação do SDK (docs.higgsfield.ai) e a página de cada modelo na
   consola (parâmetros e preço) antes de escrever `tools/higgsfield.py`: upload das fotos
   com `higgsfield_client.upload`, pedido com `subscribe`, download do resultado. O
   `subscribe` devolve o estado final sem lançar erro: só `status == "completed"` é sucesso.

## Teste

Orçamento: o saldo da API (7,29 USD a 6 de outubro de 2026). Registar o custo de cada pedido.

1. Upscale das 5 fotos a 4K (Bytedance Image Upscale). Comparar recortes 1:1 com o
   original: caixilhos, ripado, juntas do chão. Re-render do corte cinematográfico a
   3840x2160 a partir das fotos 4K (cerca de 4 vezes o tempo do render a 1080p).
2. Dois planos de 5 s, 1080p, sem som, com Kling 3.0 e com Seedance 2.5:
   - `04_pool_view`: avanço lento para a janela, água e folhagem a mexer.
   - `03_kitchen_aisle`: avanço lento ao longo da ilha.

   Prompt base: câmara gimbal estável, horizonte nivelado, sem roll, avanço lento,
   sem pessoas, sem objetos novos, arquitetura e materiais inalterados.
3. Um movimento contínuo com `02_kitchen` como imagem inicial e `03_kitchen_aisle`
   como imagem final (Kling 3.0).

## Verificação de cada clip

- O primeiro fotograma coincide com a foto (e o último com a foto final, no ponto 3).
- Caixilhos e ripas direitos, com o mesmo número do princípio ao fim.
- Nada aparece nem desaparece. A vista pelas janelas é a da foto.
- Roll e tilt perto de zero, medidos com `tools/ref_shots.py`.
- Folha de contacto a 2 imagens por segundo para revisão humana.

## Motor

Novo tipo de plano `clip` em `fpv/engine.py`: vídeo como fonte, com entrada, saída e
velocidade, e o mesmo grade, cortes no tempo, gráficos e cartão final. Serve também
para filmagem real com telemóvel em gimbal.

## Entrega

- Corte cinematográfico 4K só com fotos (upscale).
- Versão com os planos gerados que passarem, lado a lado com a atual.
- Tabela: clip, modelo, custo, passou ou não, e porquê.
