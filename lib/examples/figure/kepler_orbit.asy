size(160);
defaultpen(linewidth(0.7));

pen orbitPen = linewidth(0.8);
pen axisPen = linewidth(0.5);

real a = 3.0;
real b = 2.0;
pair center = (1.0, 0.0);
path orbit = shift(center) * scale(a, b) * unitcircle;
pair sun = (0.0, 0.0);
pair body = (2.55, 1.70);

draw(orbit, orbitPen);
draw((-0.9, 0.0) -- (4.5, 0.0), axisPen);
draw((0.0, -2.5) -- (0.0, 2.6), axisPen);
draw((sun -- body), orbitPen);
draw(arc(sun, 0.72, 0, 34), axisPen);

fill(circle(sun, 0.10), black);
fill(circle(body, 0.08), black);

label("$O$", sun, SW);
label("$m$", body, NE);
label("$r$", 0.55 * body, NW);
label("$\theta$", 0.8 * dir(5), NE);
label("$x$", (4.5, 0.0), E);
label("$y$", (0.0, 2.6), N);
