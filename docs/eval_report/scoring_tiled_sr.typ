#set page(
  paper: "a4",
  margin: (top: 16mm, bottom: 15mm, left: 17mm, right: 17mm),
  footer: context {
    set text(size: 8pt, fill: rgb("#78716c"))
    grid(
      columns: (1fr, auto),
      [Scoring a tiled enlargement],
      [26 October 2026  ·  #counter(page).display() of #counter(page).final().first()],
    )
  },
)
#set text(font: "Libertinus Serif", size: 10.4pt, fill: rgb("#1c1917"), lang: "en")
#set par(justify: true, leading: 0.52em, spacing: 0.52em)
#show figure.caption: set par(justify: false, leading: 0.42em)
#set heading(numbering: none)
#show heading.where(level: 1): it => {
  set text(font: "Inter", size: 18pt, weight: "semibold", fill: rgb("#1e3a5f"))
  block(below: 0.15em, it.body)
}
#show heading.where(level: 2): it => {
  set text(font: "Inter", size: 11.5pt, weight: "semibold", fill: rgb("#1e3a5f"))
  block(above: 0.85em, below: 0.28em, it.body)
}
#show heading.where(level: 3): it => {
  set text(font: "Inter", size: 10.4pt, weight: "semibold", fill: rgb("#292524"))
  block(above: 0.55em, below: 0.15em, it.body)
}

#block(below: 0.9em)[
  #text(font: "Inter", size: 9pt, fill: rgb("#57534e"))[
    An evaluation design for tiled super-resolution. Written so the terms are defined as they appear.
  ]
]

= Scoring a tiled enlargement

A super-resolution model takes a small photograph and paints a much larger one. At the sizes we care about, about thirty megapixels and up, the large image does not fit through the model in one pass. The model paints it as overlapping squares, and those squares are blended into one picture. The squares are the *tiles*.

The model itself is a separate piece of work. This note is the score that says whether the large picture is acceptable, and that says so for a reason we can point at. A single distance to the original photograph is not that score. A slightly blurry tile can sit closer, pixel for pixel, than a sharp tile whose grass has moved by a few pixels. Two neighbouring tiles can each look plausible and still be a failure, if one invented sharp brick and the other invented a blur.

== Two questions

Every score we keep answers one of two questions.

*Without the original large photograph.* We have the small input and the large result. Did the result keep the colour and the shading that the input had already fixed? And do tiles of the same material, the same wall or the same foliage, invent the same kind of detail?

*With the original.* Where the original has a real edge, a corner, or a piece of text, did the result keep that edge in place? Where the original is texture, did the result keep the look of that texture, without demanding the same pixels? If the result left the original, did the tiles leave it together, or did neighbouring tiles leave it in different ways?

That second question separates four situations that one distance number folds together. A tile can match the original. It can leave the original while its neighbours leave with it, as in a uniform sharpening or a uniform colour cast. It can leave the original while its neighbours do something else, which is a tiling failure. Or it can match while its neighbours do not, an island of a different treatment inside an otherwise faithful image.

== Why the image is split by scale

An enlargement by eight is free to invent only the detail finer than the small photograph could have held. Everything coarser was already in the input. One comparison for both parts makes the two questions fight. A sharp invented texture gets charged as a wrong pixel, and a drift in colour can hide inside a network trained to ignore colour. The score therefore treats each part of the picture on its own terms.

#block(width: 100%, above: 0.35em, below: 0.2em)[
  #set text(size: 9.2pt)
  #set par(justify: false, leading: 0.42em)
  #table(
    columns: (32mm, 1fr),
      stroke: (x, y) => (
        left: none, right: none,
        top: if y == 0 { 0.6pt + rgb("#1e3a5f") } else { 0.3pt + rgb("#e7e5e4") },
        bottom: if y == 0 { 0.4pt + rgb("#1e3a5f") } else { none },
      ),
      inset: (x: 4pt, y: 3.4pt),
      fill: (x, y) => if y == 0 { rgb("#f5f5f4") } else { none },
      table.header(
        [*Part of the picture*], [*What a good result does there*],
      ),
    [Coarse colour],
    [Matches the small input. With the original in hand, matches the original's coarse colour.],
    [Tile boundary],
    [No extra step. A step of one pixel is a seam. A network that looks at patches of 8 or 14 pixels averages a seam away, so the seam is measured on the pixels themselves.],
    [Finest speckles],
    [No regular grid, ringing, or checkerboard. A missing fine band is blur, and blur is scored with the middle detail.],
    [Invented detail],
    [The band the model was free to invent. Judged as a texture: how much, in which direction, of what colour. Position is enforced only on a locked edge.],
    [One material],
    [The same kind of detail in every such tile, including tiles far apart. Two tiles equally far from the original are still a failure if one is sharp and the other is soft.],
  )
]

== The pixel score

The first version described each tile by its colour and its texture, and then asked tiles of the same material to come out identical. A photograph does not work that way. Grass in the sun and grass in the shade are one material and two textures. An output that invents little, and invents it evenly, therefore scored above the photograph. On fifty photographs held out from every fit, a plain bicubic enlargement scored about 62 and the photograph scored about 17 to 21.

The current score keeps the model's own tile grid and changes the comparison. Each tile is described only by the detail above the input's limit: the amount of energy in each frequency band and each direction, whether that texture has one dominant direction, how much of the detail is colour rather than brightness, and whether the finest band is a spike, such as ringing or a checkerboard, rather than a smooth falloff.

*Without the original,* the description a natural tile would have is predicted from the matching tile of the small input. The prediction is a fit on pairs of real photographs and their reduced copies, so it is the scale behaviour of photographs, measured on those pairs. A tile is charged only when it leaves that prediction by more than three things at once: the spread already seen in photographs, a change large enough to see, and the uncertainty of estimating a spectrum on a small square. Four numbers come out of this.

- *Detail.* Too little invented texture, too much, false colour, or a periodic pattern.
- *Consistency.* Tiles of one material that depart in different directions. A departure shared by the whole image, such as everything being a little sharp, counts as detail.
- *Coarse colour,* against the input, after subtracting the small mismatch a real photograph already has with a reduced copy of itself.
- *Seams.* A step along the tile boundary, beyond the step in the rows beside it, and beyond any step the input already has on that line.

Which tiles count as one material is decided from the small input alone. The result cannot move the grouping it is judged by.

*With the original,* the target for each tile is that same tile of the original. The same ideas apply, plus three that only an original makes possible. Coarse colour is compared with the original. The mix of colours in the tile is compared as a distribution, so the flecks of a texture need not land on the same pixels. Structure compares edges, and only where the original's edge is locked: its fine version and its coarse version line up. A roofline and a letter lock. Foliage and noise do not, and there a shift of a few pixels is free. A second number says whether a departure is shared across tiles, near one, or different from tile to tile, near a half. Each tile also gets one sentence naming its largest error, for example too little horizontal detail in the finest band.

#figure(
  block(width: 100%)[
    #set text(size: 9.2pt)
    #set par(justify: false)
    #table(
      columns: (1.15fr, 1fr, 0.85fr, 0.85fr),
      align: (left, left, right, right),
      stroke: (x, y) => (
        left: none, right: none,
        top: if y == 0 { 0.6pt + rgb("#1e3a5f") } else { 0.3pt + rgb("#e7e5e4") },
        bottom: if y == 0 { 0.4pt + rgb("#1e3a5f") } else { none },
      ),
      inset: (x: 4pt, y: 3.4pt),
      fill: (x, y) => if y == 0 { rgb("#f5f5f4") } else { none },
      table.header(
        [*Fifty held-out photographs*], [*What the score sees*], [*Photograph*], [*Plain enlargement*],
      ),
      [Eight times], [small input only], [94], [51],
      [Eight times], [original as well], [100], [13],
      [Four times], [small input only], [97], [52],
      [Four times], [original as well], [100], [17],
    )
  ],
  caption: [
    Higher is better, and 100 is a clean score. The analysis tile is 256 pixels on the large image. With the original, the photograph beat every damaged version on every image. A damaged version here means a plain enlargement, an over-sharpened enlargement, texture with its positions scrambled, detail that changes from tile to tile, a colour shift that flips between tiles, or tiles pasted two pixels out of place. Without the original, a legal shift of texture, moved only where the original has no locked edge, ties the photograph. That tie is the intended result: without the original, two natural textures of one material cannot be told apart.
  ],
)

The earlier score that used the original returned zero for a plain enlargement, a legal texture shift, and a uniform sharpening alike, so those three could not be ranked. On a three-megapixel photograph the current score takes about one second without the original and about three seconds with it, on one processor. The same pixel work runs on a graphics processor when one is given. At thirty-three megapixels the estimate is a little over ten seconds and about half a minute. The earlier score took several minutes.

== What a frozen network may add

A frozen network is one we do not train and do not change. It turns a tile into a description that a few colour and spectrum numbers do not carry. It is an addition to the pixel score. Three jobs belong to it.

*Which tiles are the same stuff.* Colour calls a blue sky and a blue wall one material, and it splits one wall when the light changes. A network trained to describe image content, run on each tile, separates those cases. With the original, that description is taken from the original. Without it, the description is taken from a plain enlargement of the small input. It is never taken from the result, so a method cannot retouch the judge.

*Texture as a cloud of features.* The middle layers of a standard vision network, the same family used by the usual perceptual distances, carry more texture than the pixel summary. Each tile becomes a cloud of those features, and the clouds are compared as clouds. A point-by-point comparison is added only on locked edges. Comparing every feature pixel would charge grass for failing to land on the same pixels.

*A change of content,* and only when the original is available. A coarser layer of that content network moves when a tile has invented a different object, rather than another sample of the same texture.

Pixels stay responsible for coarse colour, for the one-pixel seam, and for the check on the finest speckles.

The feature score without an original began with the same flaw as the first pixel score. Inventing nothing is perfectly consistent, so a plain enlargement was the optimum. We now fit, on photograph pairs, how much feature energy a natural enlargement adds in each layer, and how much tiles of one material normally disagree. Missing detail then costs, and only disagreement beyond the natural amount costs. On a controlled picture the feature score ranks the photograph above a plain enlargement, and both of them above a clash in which half the tiles stay sharp and half are softened.

A bank of ordinary filters, with no learned weights, runs the same comparison. It is the control. If the learned networks do not beat those filters on the damage tests below, the gain came from comparing distributions across tiles, and from the pixel terms, rather than from what the networks were trained on.

One rule sits on top of this. A network that a method was built with is not a silent judge of that method. If a method reconstructs its image through a particular autoencoder, that autoencoder's features are left out for that method. If the method was trained alongside the content network, those terms are marked and the run is repeated with a different network in their place.

#pagebreak()

== What is settled, and what is still a run

The two questions are implemented, each as its own score, and the encoder layer sits on both. Every piece of a score is kept separately, with equal weight for now. The size of "a visible change" is a conventional value, a step in colour of about 1.5 units, a turn of about four degrees in a strongly oriented texture, and so on. It is not a fit to human judgements. The public sets of human comparisons are small patches and contain no tile seams and no distant clash of texture, so they cannot set the consistency weights. Those weights need a study on crops that show several tiles of one enlargement. Until then the pieces can be recombined without scoring the images again.

The fit shipped with the code was made on fifty images from a standard photograph set, reduced by the same successive bicubic shrinking used for our inputs. Before a table is quoted, that fit should be repeated on the large set we will publish, at the tile sizes of the experiments, and on images that will not themselves be scored.

The remaining measurement is the full table. It covers the original, a plain enlargement, and the methods under test, with and without the original, on the tile sizes those methods were run with. The encoder comparison is the same images plus a fixed set of damages, each of which should move only the term that owns it: a legal shift of texture should barely move the score, a smooth colour ramp should move coarse colour, a two-pixel misregistration should move the seam, a loss of detail on half the tiles should move consistency, and a checkerboard of small colour offsets should move both colour and seams.

== How to read one image

With the original, each tile lands in one of four cells. The two axes are whether that tile still matches the original, and whether its neighbours departed in the same way. A pixel error sees only the rows. The bottom-left cell is the case it misses: every tile is some distance from the original, and the picture is still coherent.

#block(width: 100%, above: 0.3em)[
  #set text(size: 9.2pt)
  #set par(justify: false)
  #table(
    columns: (1.15fr, 1fr, 1fr),
    stroke: (x, y) => (
      left: none, right: none,
      top: if y == 0 { 0.6pt + rgb("#1e3a5f") } else { 0.3pt + rgb("#e7e5e4") },
      bottom: if y == 0 { 0.4pt + rgb("#1e3a5f") } else { none },
    ),
    inset: (x: 4pt, y: 3.6pt),
    fill: (x, y) => if y == 0 { rgb("#f5f5f4") } else { none },
    table.header([ ], [*Neighbours agree*], [*Neighbours disagree*]),
    [*This tile matches*], [Faithful], [An island],
    [*This tile has left*], [A shared departure], [A tiling failure],
  )
]

At eight times, with only the small input, the photograph in the table scores 94 and a plain enlargement scores 51. The enlargement invented no detail, and missing detail now costs. With the original the same enlargement scores 13, because the missing detail can be compared with the photograph's own detail rather than with the range of natural photographs. The legal texture shift stays near 88: the look was kept, the pixels moved, and the locked edges were left alone.

#block(
  width: 100%,
  inset: (x: 9pt, y: 7pt),
  fill: rgb("#f5f5f4"),
  stroke: (left: 2pt + rgb("#1e3a5f")),
  above: 0.7em,
)[
  #set par(justify: false, spacing: 0.4em)
  #text(font: "Inter", size: 10pt, weight: "semibold", fill: rgb("#1e3a5f"))[Where the pieces live]
  #set text(size: 9.4pt)
  - Pixel score, WCSI-2: `genet/wcsi2.py`. Derivation and the full held-out table: `genet/WCSI2.md`.
  - Frozen networks on that score: `with_encoder/`. The weight-free filter bank is the control.
  - A directory of images, one worker per graphics processor: `genet_ablation/`.
]
