"""Function words of the six languages most likely in questions (EN, PT, ES, DE, FR, IT), never used as rare terms.

A function word of a language the corpus barely uses occurs in few chunks of the tenant (Portuguese `em` once, in an
English glossary where it means "electromagnetic"), so a document-frequency rule alone takes it for a rare term and
pulls unrelated chunks into every question in that language. The words are written as the full-text index stores
them: lowercased and unaccented, apostrophe-split (`dell'olio` gives `dell` and `olio`). Content words are left out
on purpose, even common ones: rarity already handles those.
"""

ENGLISH = """
a an the and or of to in on at by for from with without is are was were be been being do does did have has had it its
this that these those what which who whom whose when where why how i my me we our you your he she they them their
there here can could should would will shall may might must not no if then than as into about any all each every some
such much many more most other also only so
"""

PORTUGUESE = """
a o as os um uma uns umas de do da dos das em no na nos nas num numa por pelo pela pelos pelas para pra com sem sob
sobre entre ate e ou que qual quais quanto quanta quantos quantas quando onde como porque se nao sim ao aos ha eu voce
ele ela eles elas meu minha seu sua seus suas este esta estes estas esse essa esses essas isso isto aquele aquela
neste nesta nesse nessa deste desta desse dessa deve devo devem ser sao foi tem ter mais menos muito cada todo toda
todos todas tambem ja so apos
"""

SPANISH = """
el la los las lo un una unos unas de del al en por para con sin sobre entre hasta desde y o u que cual cuales cuanto
cuanta cuantos cuantas cuando donde como se no si es son esta estan ser hay mi su sus tu este esta estos estas ese
esa eso debo debe deben cada mas muy todo todos tambien ya
"""

GERMAN = """
der die das den dem des ein eine einer eines einem einen und oder aber von vom zu zum zur im in an am auf aus bei mit
nach fur uber unter vor durch gegen ohne um bis ist sind war wird werden wurde hat haben kann konnen muss mussen soll
sollte darf wie was welche welcher welches welchen wann wo warum wieviel nicht kein keine es er sie ich wir ihr sich
als auch noch nur so dass wenn ob je jede jeder jedes alle
"""

FRENCH = """
le la les l un une des du de d au aux a en dans par pour avec sans sur sous entre jusqu et ou que qu qui quel quelle
quels quelles quoi quand comment combien pourquoi ce cet cette ces est sont etre faut il elle ils elles on je nous
vous se ne pas plus moins tout tous toute toutes chaque son sa ses leur leurs mon ma mes
"""

ITALIAN = """
il lo la i gli le l un uno una di del dello della dei degli delle dell a al allo alla ai agli alle all da dal dalla
dai dall in nel nello nella nei negli nelle nell su sul sulla sull con per tra fra e ed o che chi quale quali quanto
quanta quanti quante quando dove come perche se non si sono essere ogni ciascun ciascuno piu meno tutto tutti deve
devo devono mio suo sua suoi
"""

STOPWORDS = frozenset(word for words in (ENGLISH, PORTUGUESE, SPANISH, GERMAN, FRENCH, ITALIAN) for word in words.split())
