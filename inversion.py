from manim import *
import numpy as np
import cmath

def dist(A, B):
    return np.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)

def line(A,B,xlim,ylim):
    if B[0]<A[0]:
        m,n = line(B, A, xlim, ylim)
        return n,m
    elif A[0]==B[0] :
        if A[1]<B[1] :
            return np.array([A[0], ylim[0], 0]), np.array([A[0], ylim[1], 0])
        elif A[1]>B[1] :
            return np.array([A[0], ylim[1], 0]), np.array([A[0], ylim[0], 0])
        else :
            return np.array([[0,0,0],[1,0,0]])
    else :
        a = (B[1]-A[1])/(B[0]-A[0])
        b = A[1]-a*A[0]
        X = np.array([xlim[0], xlim[0] * a + b, 0])
        Y = np.array([xlim[1], xlim[1] * a + b, 0])
        return X,Y

class Inverse:

    def __init__(self, center, radius: float):
        self.c = center
        self.r = radius
        self.ptReg = []

    def is_in(self, P):
        d = dist(self.c, P)

        if d < self.r:
            return 1
        elif d == self.r:
            return 0
        else:
            return -1

    def p_inv(self, P):
        self.ptReg.append(P)
        case = self.is_in(P)
        if case == 0:
            return P
        elif (P == self.c).all() :
            return (float('inf'), float('inf'))
        else:
            v = (P[0] - self.c[0], P[1], self.c[1])
            d1 = dist(P, self.c)
            v = (v[0] / d1, v[1] / d1)
            d2 = self.r ** 2 / d1
            v = (v[0] * d2, v[1] * d2)
            P2 = np.array([self.c[0] + v[0], self.c[1] + v[1], 0])
            self.ptReg.append(P2)
            return P2


xlim = np.array([-7, 7, 0])
ylim = np.array([-4, 4, 0])
center = np.array([0,0,0])
r = 1
vib_green = rgb_to_color([0.1,0.8,0.1])

coordA = np.array([-0.4856268383057753, -0.874166216412609, 0])
coordB = np.array([-0.9668208501760046, 0.25545536530859503, 0])
coordC = np.array([0.6564943575059553, 0.7543309343801585, 0])
coordD = np.array([0.33260894816289105, -0.9430648374327055, 0])
coords = np.array([coordA, coordB, coordC, coordD])
angles = [4.205308885483417, 2.8832739872311213, 0.8546343728202224, 5.05145766759142]

class inversion(MovingCameraScene):
    def construct(self):

        self.camera.frame.scale(1 / 3)

        title = Tex("Ptolemy's Theorem")
        self.play(Write(title))
        self.wait()
        self.play(Unwrite(title, reverse=False))
        self.remove(title)

        self.wait()

        circle = Circle(radius=r, stroke_width=2, color=BLUE, z_index=-4).move_to(center)
        self.play(GrowFromCenter(circle))

        figure1 = self.create_figure(circle)
        self.wait()
        formula = self.show_formula(figure1)
        self.wait(2)
        self.special_case(figure1, formula)

        self.play(self.camera.frame.animate.move_to(center))

        title = Tex("Inversion")
        self.play(Write(title))
        self.wait()
        self.play(Unwrite(title, reverse=False))
        self.remove(title)

        self.wait()

        O = Dot(center, radius=0.03, color=BLUE, z_index=-1)
        circle_g = Group(circle, O)
        self.play(Create(O), FadeIn(circle))

        self.play(self.camera.frame.animate.scale(3))

        #for the entire plane

        grid, inner_grid = self.all_plane(circle)
        self.wait(5)
        self.play(Uncreate(grid, lag_ratio=0.8), Uncreate(inner_grid, lag_ratio=0.8))
        self.remove(grid, inner_grid)

        self.play(self.camera.frame.animate.scale(1/3).move_to(RIGHT))
        self.camera.frame.save_state()
        self.wait()

        #first constructions

        G1 = self.one_point(np.array([2, 0, 0]), circle)
        self.animate_fig(G1, pace=-1)
        self.wait()
        self.clean(G1)
        self.wait()
        G2 = self.one_point(np.array([0.4, 0, 0]), circle)
        self.animate_fig(G2, pace=-1, reversed=True)
        self.wait()
        self.clean(G2)

        self.wait()
        self.play(self.camera.frame.animate.scale(3/2))

        #moving the point around

        path = np.array([[2, 0, 0], [1.5, 1.5, 0], [4, 0, 0], [1.75, -1.5, 0], [1, -0.5, 0]])
        P = self.Bezier_from_path(path)
        f = self.animate_path(P, circle)
        self.wait()
        self.clean(f)

        self.play(Restore(self.camera.frame))
        self.wait()

        #special cases
        S1 = self.one_point(np.array([1, 0, 0]), circle)
        self.animate_fig(S1, pace=-1.5)

        self.clean(S1)
        self.wait()

        S2 = Line(np.array([2,0,0]),np.array([100, 0, 0]))
        f_S2 = self.animate_path(S2, circle, time=20)
        self.wait(3)
        self.clean(S2, f_S2)
        self.wait()

        #first formula

        G1 = self.one_point(np.array([2, 0, 0]), circle)
        self.animate_fig(G1, pace=1)
        self.wait()
        self.formula(G1, circle_g)

        #second formula

        self.play(self.camera.frame.animate.move_to([0, -0.8, 0]))
        line_t = (np.array([-7, -4, 0]), np.array([0, -1.2, 0]))
        f, i, l = self.image_of_line(line_t, circle)
        self.formula_line(line_t, circle_g, i, l)

        #final part

        self.final_figure(circle_g)

        self.wait(3)

    def clean(self, *M):
        self.play(FadeOut(*M))
        self.remove(*M)

    def limits(self):
        xlim = (self.camera.frame_center[0] - self.camera.frame_width / 2, self.camera.frame_center[0] + self.camera.frame_width / 2)
        ylim = (self.camera.frame_center[1] - self.camera.frame_height / 2, self.camera.frame_center[1] + self.camera.frame_height / 2)
        return (xlim, ylim)

    def elements_OP(self, point):
        R = (r ** 2) / dist(center, point)
        alpha_x = (np.arccos(R)+cmath.polar(complex(point[0], point[1]))[1])%(2*np.pi)
        x2 = R * ((point - center) / dist(point, center))
        t = np.array([np.cos(alpha_x), np.sin(alpha_x), 0])
        m, n = line(center, point, (-7, 7), (-4, 4))
        return (point,t,x2,m,n),alpha_x

    def one_point(self, point, circle, reversed=False):

        if dist(point, center)<r and not reversed :
            Inv = Inverse(center, r)
            return self.one_point(Inv.p_inv(point), circle, reversed=True)

        points, alpha_x = self.elements_OP(point)

        x,t,x2,m,n = points

        X = Dot(x, color=RED, radius=0.05, z_index=-1)
        L = Line(circle.get_center(), n, stroke_width=1, color=GRAY, z_index=-5)
        T = TangentLine(circle,
                        alpha=alpha_x / (2 * np.pi),
                        length=2*dist(np.array([xlim[0],ylim[0]]),np.array([xlim[1],ylim[1]])),
                        stroke_width=1,
                        color=GRAY,
                        z_index=-5)
        Xt = Dot(t, color=GRAY, radius=0.05, z_index=-1)
        X2 = Dot(x2, color=PURPLE, radius=0.05, z_index=-1)
        H = Line(t, x2, color=GRAY, stroke_width=1, z_index=-5)

        return VGroup(X,L,T,Xt,H,X2)

    def animate_fig(self, figure, reversed = False, pace = 0):
        '''
        :param pace: -1 lent, 0 normal, 1 rapide
        :return:
        '''
        A = ()
        X,L,T,Xt,H,X2 = figure
        if not reversed :
            A += (Create(X),)
            A += (Create(L),)
            A += (GrowFromPoint(T, X.get_center()),)
            A += (Create(Xt),)
            A += (Create(H),)
            A += (Create(X2),)
        else :
            A += (Create(X2),)
            A += (Create(L),)
            A += (Create(H),)
            A += (Create(Xt),)
            A += (GrowFromPoint(T, Xt.get_center()),)
            A += (Create(X),)

        if pace > 0 :
            self.play(*A)
        else :
            for a in A :
                if pace <= 0 :
                    self.play(a)
                    if pace < 0 :
                        self.wait(-pace)

    def Bezier_from_path(self, path):
        B = []
        n = len(path)
        for i in range(n):
            u = (path[(i+1)%n]-path[i-1])/2
            v = (path[(i+2)%n]-path[(i)])/2
            B.append(CubicBezier(path[i], path[i]+u, path[(i+1)%n]-v, path[(i+1)%n]))

        P = VMobject()
        for b in B:
            P.add_subpath(b.points)

        return P

    def animate_path(self, P, circle, figure=None, time=0):

        if figure==None :
            figure = self.one_point(P.points[0], circle)
            self.animate_fig(figure)

        if time==0 :
            time = P.get_arc_length(2) * 2

        def figure_updater(G):
            G2 = self.one_point(G[0].get_center(), circle)
            for i in range(len(G)):
                G[i].become(G2[i])

        self.play(MoveAlongPath(figure[0], P, rate_func=rate_functions.ease_in_out_sine),
                  UpdateFromFunc(figure, figure_updater),
                  run_time=time)

        return figure

    def image_of_line(self, line_t, circle):
        A,B = line_t
        cam = self.camera
        xlim, ylim = self.limits()
        M, N = line(A, B, xlim, ylim)
        path = Line(M,N)
        figure = self.one_point(path.points[0], circle)
        self.animate_fig(figure, pace=1)

        I = VMobject(color=PURPLE, stroke_width=2.5, z_index=-2)
        I.set_points_as_corners([figure[5].get_center(), figure[5].get_center()])
        L = VMobject(color=RED, stroke_width=2.5, z_index=-2)
        L.set_points_as_corners([figure[0].get_center(), figure[0].get_center()])

        def update_path(path, dot):
            previous_path = path.copy()
            previous_path.add_points_as_corners([dot.get_center()])
            path.become(previous_path)

        I.add_updater(lambda p : update_path(p, figure[5]))
        L.add_updater(lambda p : update_path(p, figure[0]))

        self.add(I, L)
        self.animate_path(path, circle, figure=figure, time=7)
        self.clean(figure)

        circle_I = Circle.from_three_points(I.points[len(I.points)//3], I.points[2*len(I.points)//3], center,
                                            color=PURPLE, stroke_width=2.5, z_index=-2)

        self.play(FadeIn(circle_I), FadeOut(I))
        self.remove(I)

        return figure, circle_I, L

    def formula(self, figure, circle_g):

        X, L, T, Xt, H, X2 = figure

        def T_updater(T : VGroup):
            for i in range(len(T[0])-1):
                T[i+1].move_to(T[0][i].points[0])

        T1_1 = Line(X.get_center(), Xt.get_center(), color=YELLOW, stroke_opacity=0.6, stroke_width=2)
        T1_2 = Line(Xt.get_center(), center, color=YELLOW, stroke_opacity=0.6, stroke_width=2)
        T1_3 = Line(center, X.get_center(), color=RED, stroke_opacity=0.6, stroke_width=2)
        T1_R = RightAngle(T1_1, T1_2, T1_1.get_arc_length(2) / 10, quadrant=(-1, 1), stroke_width=1, z_index=-2)

        T2_1 = Line(X2.get_center(), Xt.get_center(), color=YELLOW, stroke_opacity=0.6, stroke_width=2)
        T2_2 = Line(Xt.get_center(), center, color=YELLOW, stroke_opacity=0.6, stroke_width=2)
        T2_3 = Line(center, X2.get_center(), color=RED, stroke_opacity=0.6, stroke_width=2)
        T2_R = RightAngle(T2_1, T2_3, T2_1.get_arc_length(2) / 8, quadrant=(1, -1), stroke_width=1, z_index=-2)

        T1 = VGroup(VGroup(T1_1, T1_2, T1_3, T1_R),
                    LabeledDot("A", point=X.get_center()).scale(0.3),
                    LabeledDot("T", point=Xt.get_center()).scale(0.3),
                    LabeledDot("O", point=center).scale(0.3))

        T2 = VGroup(VGroup(T2_1, T2_2, T2_3, T2_R),
                    LabeledDot("A'", point=X2.get_center()).scale(0.27),
                    LabeledDot("T", point=Xt.get_center()).scale(0.3),
                    LabeledDot("O", point=center).scale(0.3))

        T1.add_updater(T_updater)
        T2.add_updater(T_updater)

        T1_bis = T1.copy()
        T2_bis = T2.copy()

        self.play(Create(T1), Create(T2), run_time=4)
        self.wait(3)

        Op1 = (T1[0][i].animate.set_opacity(1) for i in range(3))
        Op2 = (T2[0][i].animate.set_opacity(1) for i in range(3))


        self.play(FadeOut(figure, circle_g), *Op1, *Op2)
        self.remove(figure, circle_g)

        self.wait()

        self.play(T1[0].animate.shift(DOWN))
        self.play(T2[0].animate.rotate(-np.pi/2).shift(0.3*UP))
        self.play(T1[0].animate.apply_function(lambda X : np.array([X[0], -X[1], X[2]])).move_to(T2.get_center()+1.5*DOWN).rotate(-np.pi/6))

        self.wait()

        Alpha1 = Angle(T1[0][2], T1[0][0], dist(T1[1].get_center(), T1[3].get_center()) / 6, quadrant=(-1, 1), stroke_width=1, z_index=-5)
        Alpha2 = Angle(T2[0][1], T2[0][0], dist(T2[1].get_center(), T2[3].get_center()) / 2, quadrant=(1, -1), stroke_width=1, z_index=-5)
        text1 = MathTex(r'\alpha').next_to(Alpha1.get_center(), direction=0.2*LEFT).scale(0.3)
        text2 = text1.copy().next_to(Alpha2.get_center(), direction=0.3*LEFT)
        self.play(Create(Alpha1), Create(Alpha2), Write(text1), Write(text2))
        self.wait()

        formula1 = MathTex(r"sin({{\alpha}})={{\frac{OA'}{OT}}}").scale(0.3).move_to(2*RIGHT+0.5*UP)
        formula2 = MathTex(r"sin({{\alpha}})={{\frac{OT}{OA}}}").scale(0.3).move_to(2*RIGHT+0.5*DOWN)

        self.play(Write(formula1), Write(formula2))

        self.wait()

        frac_1 = formula1.submobjects[3].copy()
        frac_2 = formula2.submobjects[3].copy()
        formula = MathTex(r"{{ \frac{OA'}{OT} }}={{ \frac{OT}{OA} }}").scale(0.3).move_to(2*RIGHT)

        self.add(frac_1, frac_2)
        self.remove(formula1.submobjects[3], formula2.submobjects[3])
        self.play(FadeOut(formula1), FadeOut(formula2))
        self.play(frac_1.animate.move_to(formula.submobjects[0].get_center()),
                  frac_2.animate.move_to(formula.submobjects[2].get_center()),
                  Write(formula.submobjects[1]))

        self.add(formula)
        self.remove(frac_1, frac_2, formula1, formula2)

        self.wait()

        OA_ = VGroup(formula.submobjects[0].submobjects[0],
                     formula.submobjects[0].submobjects[1],
                     formula.submobjects[0].submobjects[2]).copy()

        OT1 = VGroup(formula.submobjects[0].submobjects[4],
                     formula.submobjects[0].submobjects[5]).copy()

        OT2 = VGroup(formula.submobjects[2].submobjects[0],
                     formula.submobjects[2].submobjects[1]).copy()

        OA = VGroup(formula.submobjects[2].submobjects[3],
                    formula.submobjects[2].submobjects[4]).copy()

        eq = formula.submobjects[1].copy()

        final_formula1 = MathTex(r"{{OA}}\times{{OA'}}={{OT}}^2").scale(0.3).move_to(formula.get_center())

        self.add(OA_, OA, OT1, OT2, eq)
        self.clean(formula)
        self.play(OA.animate.move_to(final_formula1.submobjects[0].get_center()),
                  OA_.animate.move_to(final_formula1.submobjects[2].get_center()),
                  OT1.animate.move_to(final_formula1.submobjects[4].get_center()),
                  OT2.animate.move_to(final_formula1.submobjects[4].get_center()),
                  eq.animate.move_to(final_formula1.submobjects[3].get_center()),
                  Create(final_formula1.submobjects[5]),
                  Write(final_formula1.submobjects[1]))

        self.add(final_formula1)
        self.remove(OA_, OA, OT1, OT2, eq)

        self.wait()

        final_formula2 = MathTex(r"{{OA}}\times{{OA'}}={{r}}^2").scale(0.3).move_to(formula.get_center())

        self.play(*(final_formula1.submobjects[i].animate.become(final_formula2.submobjects[i]) for i in range(len(final_formula1.submobjects))))
        self.wait()

        self.play(FadeOut(T1, T2, Alpha1, Alpha2, text1, text2))
        T1.become(T1_bis)
        T2.become(T2_bis)
        self.play(FadeIn(figure, circle_g, *(T1[i+1] for i in range(3)), *(T2[i+1] for i in range(3))),
                  final_formula1.animate.shift(0.7*DOWN+0.4*LEFT))

        self.wait(3)

        self.play(FadeOut(final_formula1, figure, *(T1[i+1] for i in range(3)), *(T2[i+1] for i in range(3))))

    def formula_line(self, line_t, circle_g, circle_I, L):
        cam = self.camera
        circle, O = circle_g

        cam.frame.scale(2).shift(DOWN)
        xlim = (cam.frame_center[0] - cam.frame_width / 2, cam.frame_center[0] + cam.frame_width / 2)
        ylim = (cam.frame_center[1] - cam.frame_height / 2, cam.frame_center[1] + cam.frame_height / 2)
        M, N = line(line_t[0], line_t[1], xlim, ylim)
        L2 = Line(M, N, color=RED, stroke_width=2.5, z_index=-2)
        self.add(L2)
        self.remove(L)
        cam.frame.shift(UP).scale(0.5)

        self.wait(3)

        F_A = self.one_point(line_t[0] + 7 * L2.get_unit_vector(), circle)
        F_B = self.one_point(line_t[1] + L2.get_unit_vector(), circle)
        L_AB = Line(F_A[5], F_B[5], color=PURPLE, stroke_width=2.5, z_index=-2)
        self.play(FadeIn(F_A[0], F_A[1], F_A[5], F_B[0], F_B[1], F_B[5]))

        def T_updater(T: VGroup):
            for i in range(len(T[0])):
                T[i + 1].move_to(T[0][i].points[0])

        T1_1 = Line(center, F_A[5].get_center(), color=YELLOW, stroke_opacity=0.6, stroke_width=2)
        T1_2 = Line(F_A[5].get_center(), F_B[5].get_center(), color=YELLOW, stroke_opacity=0.6, stroke_width=2)
        T1_3 = Line(F_B[5].get_center(), center, color=YELLOW, stroke_opacity=0.6, stroke_width=2)

        T2_1 = Line(center, F_A[0].get_center(), color=YELLOW, stroke_opacity=0.6, stroke_width=2)
        T2_2 = Line(F_B[0].get_center(), center, color=YELLOW, stroke_opacity=0.6, stroke_width=2)
        T2_3 = Line(F_A[0].get_center(), F_B[0].get_center(), color=YELLOW, stroke_opacity=0.6, stroke_width=2)

        T1 = VGroup(VGroup(T1_1, T1_2, T1_3).set_z_index(-1),
                    LabeledDot("O", point=center).scale(0.3),
                    LabeledDot("A'", point=F_A[5].get_center()).scale(0.28),
                    LabeledDot("B'", point=F_B[5].get_center()).scale(0.28))

        T2 = VGroup(VGroup(T2_1, T2_2, T2_3).set_z_index(-1),
                    LabeledDot("O", point=center).scale(0.3),
                    LabeledDot("B", point=F_B[0].get_center()).scale(0.3),
                    LabeledDot("A", point=F_A[0].get_center()).scale(0.3))

        T1.add_updater(T_updater)
        T2.add_updater(T_updater)

        T2_bis = T2.copy()

        self.play(Create(T1[1]), Create(T1[2]), Create(T1[3]), Create(T2[1]), Create(T2[2]), Create(T2[3]), run_time=2)
        self.play(GrowFromPoint(L_AB, F_A[5].get_center()))

        self.wait(3)

        self.play(Create(T1[0]), Create(T2[0]), run_time=2)

        Op1 = (T1[0][i].animate.set_opacity(1) for i in range(3))
        Op2 = (T2[0][i].animate.set_opacity(1) for i in range(3))

        self.play(FadeOut(F_A[0]), FadeOut(F_A[1]), FadeOut(F_A[5]), FadeOut(F_B[0]), FadeOut(F_B[1]), FadeOut(F_B[5]),
                  FadeOut(L2), FadeOut(circle_I), FadeOut(circle_g), FadeOut(L_AB), *Op1, *Op2)
        self.remove(F_A, F_B, L2, circle_I, circle_g, L_AB)

        self.play(cam.frame.animate.shift(0.3 * DOWN))

        def f(x):
            return np.array([-x[0], x[1], 0])

        self.play(T2.animate.move_to(T1.get_center() + 1.3 * DOWN))
        self.play(T2[0].animate.apply_function(f).move_to(T1.get_center() + 1.3 * DOWN))
        self.play(T2[0].animate.rotate(0.5))

        self.wait()

        self.play(cam.frame.animate.shift(1.3 * RIGHT))

        formula = MathTex(r"{{ \frac{A'B'}{ OA'} }} = {{ \frac{BA} {OB} }}").scale(0.3).move_to(
            cam.frame_center + 1.1 * RIGHT)
        formula_bis = MathTex(r"{{A'B'}} = {{ \frac{ { BA } \times { OA' } }{ OB } }}").scale(0.3).move_to(
            cam.frame_center + 1.1 * RIGHT)

        A_B_ = VGroup(*tuple(formula.submobjects[0].submobjects[:4]))
        OA_ = VGroup(*tuple(formula.submobjects[0].submobjects[5:8]))
        BA = VGroup(*tuple(formula.submobjects[2].submobjects[:2]))
        OB = VGroup(*tuple(formula.submobjects[2].submobjects[3:5]))
        f_line = formula.submobjects[2].submobjects[2]
        eq = formula.submobjects[1]

        self.play(Write(formula))

        self.play(Indicate(A_B_, scale_factor=1, color=PURE_RED),
                  Indicate(BA, scale_factor=1, color=PURE_RED),
                  Indicate(T1[0][1], scale_factor=1, color=PURE_RED),
                  Indicate(T2[0][2], scale_factor=1, color=PURE_RED), run_time=3)
        self.play(Indicate(OA_, scale_factor=1, color=PURE_RED),
                  Indicate(OB, scale_factor=1, color=PURE_RED),
                  Indicate(T1[0][0], scale_factor=1, color=PURE_RED),
                  Indicate(T2[0][1], scale_factor=1, color=PURE_RED), run_time=3)

        A = ()

        A += (A_B_.animate.become(formula_bis.submobjects[0]),)
        A += (OA_.animate.become(VGroup(*tuple(formula_bis.submobjects[2].submobjects[3:6]))),)
        A += (BA.animate.become(VGroup(*tuple(formula_bis.submobjects[2].submobjects[:2]))),)
        A += (OB.animate.become(VGroup(*tuple(formula_bis.submobjects[2].submobjects[7:9]))),)
        A += (FadeOut(formula.submobjects[0].submobjects[4]),)
        A += (Create(formula_bis.submobjects[2].submobjects[2]),)
        A += (f_line.animate.become(formula_bis.submobjects[2].submobjects[6]),)
        A += (eq.animate.become(formula_bis.submobjects[1]),)

        self.play(*A)

        self.remove(formula, A_B_, OA_, BA, OB, f_line, eq)
        self.add(formula_bis)

        self.wait(3)

        formula_A = MathTex(r"{{OA}}\times{{OA'}}={{r^2}}").scale(0.3).move_to(formula.get_center() + 0.4 * DOWN)
        formula_A_bis = MathTex(r"{{OA'}}={{ \frac{r^2}{OA} }}").scale(0.3).move_to(formula_A.get_center())

        OA, x, OA_, eq, rr = tuple(formula_A.submobjects[:5])

        self.play(formula_bis.animate.shift(0.25 * UP))
        self.play(Write(formula_A))

        self.wait()

        A = ()

        A += (OA_.animate.become(formula_A_bis.submobjects[0]),)
        A += (FadeOut(x),)
        A += (OA.animate.become(VGroup(*tuple(formula_A_bis.submobjects[2].submobjects[3:5]))),)
        A += (eq.animate.become(formula_A_bis.submobjects[1]),)
        A += (rr.animate.become(VGroup(*tuple(formula_A_bis.submobjects[2].submobjects[:2]))),)
        A += (Create(formula_A_bis.submobjects[2].submobjects[2]),)

        self.play(*A)
        self.add(formula_A_bis)
        self.remove(formula_A, OA, x, OA_, eq, rr)

        self.wait(3)

        final_form = MathTex(r"{{A'B'}} = {{ \frac{ { BA } \times { r^2 } }{ { OB } \times { OA } } }}").scale(
            0.3).move_to(cam.frame_center + RIGHT)

        A_B_ = VGroup(*tuple(formula_bis.submobjects[0]))
        OA_ = VGroup(*tuple(formula_bis.submobjects[2].submobjects[3:6]))
        BA = VGroup(*tuple(formula_bis.submobjects[2].submobjects[:2]))
        OB = VGroup(*tuple(formula_bis.submobjects[2].submobjects[7:9]))
        f_line = formula_bis.submobjects[2].submobjects[6]
        eq = formula_bis.submobjects[1]
        x = formula_bis.submobjects[2].submobjects[2]

        rr = VGroup(*tuple(formula_A_bis.submobjects[2].submobjects[:2]))
        OA_bis = VGroup(*tuple(formula_A_bis.submobjects[0]))
        OA = VGroup(*tuple(formula_A_bis.submobjects[2].submobjects[3:5]))
        f_line_bis = formula_A_bis.submobjects[2].submobjects[2]
        eq_bis = formula_A_bis.submobjects[1]

        A = ()
        A += (FadeOut(OA_, OA_bis, eq_bis, f_line_bis, run_time=0.3),)
        A += (f_line.animate.become(final_form.submobjects[2].submobjects[5]),)
        A += (A_B_.animate.become(final_form.submobjects[0]),)
        A += (BA.animate.become(VGroup(*tuple(final_form.submobjects[2].submobjects[:2]))),)
        A += (OB.animate.become(VGroup(*tuple(final_form.submobjects[2].submobjects[6:8]))),)
        A += (eq.animate.become(final_form.submobjects[1]),)
        A += (x.animate.become(final_form.submobjects[2].submobjects[2]),)
        A += (Write(final_form.submobjects[2].submobjects[8]),)
        A += (rr.animate.become(VGroup(*tuple(final_form.submobjects[2].submobjects[3:5]))),)
        A += (OA.animate.become(VGroup(*tuple(final_form.submobjects[2].submobjects[9:11]))),)

        self.play(*A)
        self.add(final_form)
        self.remove(formula_A_bis, formula_bis, A_B_, OA_, BA, OB, f_line, eq, x, rr, OA_bis, OA, f_line_bis, eq_bis)

        self.wait(3)

        self.play(FadeOut(T1, T2))
        T2.become(T2_bis)
        self.play(FadeIn(circle_g, circle_I, L2, L_AB,
                         F_A[0], F_A[1], F_A[5], F_B[0], F_B[1], F_B[5],
                         T1[1], T1[2], T1[3], T2[1], T2[2], T2[3]), run_time=1.5)

        self.wait(5)

        self.play(FadeOut(circle_I, L2, F_A[0], final_form, L_AB, circle_g,
                          F_A[1], F_A[5], F_B[0], F_B[1], F_B[5],
                          T1[1], T1[2], T1[3], T2[1], T2[2], T2[3]))

    def final_figure(self, circle_g):

        circle, O = circle_g
        u,v = [-1.25425392, -0.38545603, 0], [-0.24169961, -1.26668224, 0]
        Inv = Inverse(center,r)
        i = Circle.from_three_points(center, Inv.p_inv(u), Inv.p_inv(v), color=PURPLE, stroke_width=1.5, z_index=-2)

        self.play(self.camera.frame.animate.scale(i.radius).move_to(i.get_center()))

        self.play(GrowFromCenter(i))
        fig = self.create_figure(i, circle_stroke_w=1.5)  # Group(circle, A, B, C, D, R, D1, D2)

        arc = Arc(r, np.pi/2, 3*np.pi/2, 9, center, color=BLUE, stroke_opacity=0.3, stroke_width=1.5, z_index=-2)

        self.play(Create(arc))
        self.play(FadeOut(arc), FadeIn(circle.set_stroke(opacity=0.1)))
        self.remove(arc)

        self.wait()

        self.play(self.camera.frame.animate.scale((2/3)/i.radius).shift(0.2*LEFT+0.2*DOWN))

        self.wait()

        F_A = self.one_point(fig[1].get_center(), circle)
        F_B = self.one_point(fig[2].get_center(), circle)
        F_D = self.one_point(fig[4].get_center(), circle)
        A_ = LabeledDot("A'", point=F_A[0].get_center()).scale(0.19)
        B_ = LabeledDot("B'", point=F_B[0].get_center()).scale(0.19)
        D_ = LabeledDot("D'", point=F_D[0].get_center()).scale(0.19)

        self.play(Create(F_A[1]), Create(F_B[1]), Create(F_D[1]))
        self.play(Create(A_), Create(B_), Create(D_))

        self.wait()

        B_D_ = Line(B_.get_center(), D_.get_center(), color=RED, stroke_width=1.5, z_index=-2)
        B_A_ = Line(B_.get_center(), A_.get_center(), color=RED, stroke_width=1.5, z_index=-2)
        A_D_ = Line(A_.get_center(), D_.get_center(), color=RED, stroke_width=1.5, z_index=-2)

        B_D_2 = B_D_.copy()
        B_A_2 = B_A_.copy()
        A_D_2 = A_D_.copy()


        self.play(GrowFromPoint(B_D_, F_B[0].get_center()))

        self.wait(3)

        self.play(self.camera.frame.animate.scale(1.5).shift(1.2*RIGHT+0.3*DOWN))

        form_line = MathTex(r"{{B'A'}}+{{A'D'}}={{B'D'}}").scale(0.3).move_to(1.4*RIGHT+1.5*DOWN)
        for t in (form_line.submobjects[i] for i in [0,2,4]) :
            t.set_color(RED)

        self.play(Transform(B_A_2, form_line.submobjects[0]))
        self.play(Transform(A_D_2, form_line.submobjects[2]), Write(form_line.submobjects[1]))
        self.play(Transform(B_D_2, form_line.submobjects[4]), Write(form_line.submobjects[3]))

        self.add(form_line)
        self.play(FadeOut(B_D_2, B_A_2, A_D_2))
        self.remove(B_A_, A_D_, B_D_2, B_A_2, A_D_2)

        self.wait()

        final_form = MathTex(r"{{A'B'}} = {{ \frac{ { BA } \times { r^2 } }{ { CB } \times { CA } } }}"
                                ).scale(0.3).move_to(form_line.get_center()+0.7*UP+0.5*RIGHT)
        self.play(Write(final_form))
        self.wait()

        final_form_BA = final_form.copy().set_opacity(0)

        final_form_AD = MathTex(r"{{A'D'}} = {{ \frac{ { DA } \times { r^2 } }{ { CD } \times { CA } } }}"
                                ).scale(0.3).move_to(final_form_BA.get_center()).set_opacity(0)
        final_form_BD = MathTex(r"{{B'D'}} = {{ \frac{ { DB } \times { r^2 } }{ { CD } \times { CB } } }}"
                                ).scale(0.3).move_to(final_form_BA.get_center()).set_opacity(0)

        big_equation = MathTex(r"{{ \frac{ { BA } \times { r^2 } }{ { CB } \times { CA } } }}"+
                               r"{{+}}"+
                               r"{{ \frac{ { DA } \times { r^2 } }{ { CD } \times { CA } } }}"+
                               r"{{=}}"+
                               r"{{ \frac{ { DB } \times { r^2 } }{ { CD } \times { CB } } }}").scale(0.3).move_to(form_line.get_center())

        self.play(*(form_line.submobjects[i].animate.move_to(big_equation.submobjects[i].get_center()) for i in range(len(form_line.submobjects))))

        self.play(FadeOut(form_line[0]), final_form_BA.submobjects[2].animate.become(big_equation.submobjects[0]))
        self.play(FadeOut(form_line[2]), final_form_AD.submobjects[2].animate.become(big_equation.submobjects[2]))
        self.play(FadeOut(form_line[4]), final_form_BD.submobjects[2].animate.become(big_equation.submobjects[4]))

        self.wait()
        self.play(Unwrite(final_form))
        self.add(big_equation)
        self.remove(final_form, final_form_AD.submobjects[2], final_form_BA.submobjects[2], final_form_BD.submobjects[2])

        self.wait()

        big_equation_2 = MathTex(r"{{ \frac{ BA }{ { CB } \times { CA } } }}" +
                                 r"{{+}}" +
                                 r"{{ \frac{ DA }{ { CD } \times { CA } } }}" +
                                 r"{{=}}" +
                                 r"{{ \frac{ DB }{ { CD } \times { CB } } }}").scale(0.3).move_to(form_line.get_center())

        def frac_transform(frac1, frac2):
            A = ()
            A += (VGroup(*(frac1.submobjects[i] for i in range(2))).animate.become(VGroup(*(frac2.submobjects[i] for i in range(2)))),)
            A += (Unwrite(VGroup(*(frac1.submobjects[i] for i in range(2,5)))),)
            A += (VGroup(*(frac1.submobjects[i] for i in range(5,11))).animate.become(VGroup(*(frac2.submobjects[i] for i in range(2,8)))),)
            return A


        self.play(*frac_transform(big_equation.submobjects[0], big_equation_2.submobjects[0]),
                  *frac_transform(big_equation.submobjects[2], big_equation_2.submobjects[2]),
                  *frac_transform(big_equation.submobjects[4], big_equation_2.submobjects[4]))

        self.wait()
        self.add(big_equation_2)
        self.remove(big_equation, big_equation.submobjects[1], big_equation.submobjects[3],
                    form_line.submobjects[1], form_line.submobjects[3])

        common_factor = MathTex(r"{{CA}} \times {{CB}} \times {{CD}}").scale(0.3).move_to(final_form.get_center())

        big_equation_3 = MathTex(r"{{BA}} \times {{CD}}+{{DA}} \times {{CB}}={{DB}} \times {{CA}}").scale(0.3).move_to(form_line.get_center()).set_z_index(1)

        self.play(Write(common_factor))
        self.wait()

        def frac_transform_2(frac, target, fact):
            f = fact.copy()
            fr = frac.copy()
            A = ()
            A += (Write(target[1]),)
            A += (VGroup(*(fr.submobjects[i] for i in range(2))).animate.become(target[0]),)
            A += (f.submobjects[2].copy().animate.become(VGroup(*(fr.submobjects[i] for i in range(3, 5))).set_opacity(0)),)
            A += (f.submobjects[0].copy().animate.become(VGroup(*(fr.submobjects[i] for i in range(6, 8))).set_opacity(0)),)
            A += (f.submobjects[4].copy().animate.become(target[2]),)
            A += (FadeOut(frac, f),)
            return A

        self.play(*frac_transform_2(big_equation_2.submobjects[0], [big_equation_3.submobjects[i] for i in range(3)], common_factor),
                  FadeOut(big_equation.submobjects[0]),
                  *frac_transform_2(big_equation_2.submobjects[2], [big_equation_3.submobjects[i] for i in range(4,7)], common_factor),
                  FadeOut(big_equation.submobjects[2]),
                  *frac_transform_2(big_equation_2.submobjects[4], [big_equation_3.submobjects[i] for i in range(8,11)], common_factor),
                  FadeOut(big_equation.submobjects[4]),
                  big_equation_2.submobjects[1].animate.move_to(big_equation_3.submobjects[3].get_center()),
                  big_equation_2.submobjects[3].animate.move_to(big_equation_3.submobjects[7].get_center()))

        self.play(Unwrite(common_factor))
        self.add(big_equation_3, Rectangle(color=BLACK,
                                           fill_color=BLACK,
                                           fill_opacity=1,
                                           height=big_equation_3.height,
                                           width=big_equation_3.width).move_to(big_equation_3.get_center()).set_z_index(0))

        self.remove(common_factor, big_equation_2)
        self.wait()
        self.play(Circumscribe(big_equation_3, stroke_width=1))
        self.wait()

        self.play(FadeOut(A_, B_, D_, circle, B_D_, F_A[0], F_A[1], F_B[0], F_B[1], F_D[0], F_D[1]))
        self.play(self.camera.frame.animate.move_to(i.get_center()+0.25*DOWN).scale(0.6),
                  big_equation_3.animate.move_to(i.get_center()+0.8*DOWN).scale(0.8))

    def all_plane(self, circle):

        grid = NumberPlane(x_range=[-20, 20, 0.5],
                           y_range=[-20, 20, 0.5],
                           z_index=-3,
                           background_line_style={
                               "stroke_color": TEAL,
                               "stroke_width": 1,
                               "stroke_opacity": 0.6
                           })
        grid.prepare_for_nonlinear_transform()

        inner_grid = NumberPlane(x_range=[-1, 1, 2 / 32],
                                 y_range=[-1, 1, 2 / 32],
                                 z_index=-3,
                                 background_line_style={
                                     "stroke_color": TEAL,
                                     "stroke_width": 1,
                                     "stroke_opacity": 0.6
                                 })
        inner_grid.prepare_for_nonlinear_transform()

        def f(X):
            if dist(X, center) <=r:
                return X
            else :
                return r*(X-center)/dist(X,center)

        inner_grid.apply_function(f)

        def I(p):
            if dist(p, center)<0.01 :
                return p
            elif p[0] == 0 or p[1] == 0:
                return p
            else:
                Inv = Inverse(center, r)
                n_p = Inv.p_inv(p)
                return np.array([n_p[0], n_p[1], 0])

        self.play(Create(grid), Create(inner_grid), run_time=2.5)
        self.wait()
        self.play(grid.animate(run_time=6).apply_function(I), inner_grid.animate(run_time=7).apply_function(I))
        self.wait()

        return grid, inner_grid

    def create_figure(self, circle, circle_stroke_w=2):

        A = LabeledDot("A").scale(0.3*circle_stroke_w/2)
        B = LabeledDot("B").scale(0.3*circle_stroke_w/2)
        C = LabeledDot("C").scale(0.3*circle_stroke_w/2)
        D = LabeledDot("D").scale(0.3*circle_stroke_w/2)

        dots = [A, B, C, D]
        for i in range(len(dots)):
            dots[i].move_to(circle.get_center()+circle.radius*coords[i])

        R = Polygon(*(d.get_center() for d in dots), color=vib_green, z_index=-1, stroke_width=3*circle_stroke_w/4)

        for d in dots:
            self.play(Create(d), run_time=0.3)
            self.wait(0.1)

        self.play(Create(R), run_time=2)

        D1 = Line(A.get_center(), C.get_center(), color=YELLOW, z_index=-1, stroke_width=3*circle_stroke_w/4)
        D2 = Line(B.get_center(), D.get_center(), color=YELLOW, z_index=-1, stroke_width=3*circle_stroke_w/4)
        self.play(Create(D1),Create(D2))

        return Group(circle, A, B, C, D, R, D1, D2)

    def show_formula(self, figure1):
        self.play(self.camera.frame.animate.shift(RIGHT))
        self.wait()

        formula = MathTex(r"{{AB}} \times {{CD}} + {{BC}} \times {{AD}} = {{AC}} \times {{BD}}").scale(0.25).shift(2.2*RIGHT)

        AB = Line(coordA, coordB, color=vib_green, z_index=-0.5, stroke_width=1.5)
        BC = Line(coordB, coordC, color=vib_green, z_index=-0.5, stroke_width=1.5)
        CD = Line(coordC, coordD, color=vib_green, z_index=-0.5, stroke_width=1.5)
        AD = Line(coordA, coordD, color=vib_green, z_index=-0.5, stroke_width=1.5)
        AC = Line(coordA, coordC, color=YELLOW, z_index=-0.5, stroke_width=1.5)
        BD = Line(coordB, coordD, color=YELLOW, z_index=-0.5, stroke_width=1.5)
        Decomp = Group(AB, CD, BC, AD, AC, BD).move_to(figure1[5])
        SubM = formula.submobjects
        j=0

        for i in range(len(SubM)):
            if i in [0,4,8] :
                if i!=8 :
                    SubM[i].set_color(vib_green)
                    SubM[i+2].set_color(vib_green)
                else :
                    SubM[i].set_color(YELLOW)
                    SubM[i+2].set_color(YELLOW)

                self.play(Transform(Decomp[j], SubM[i]), Transform(Decomp[j+1], SubM[i+2]), Write(SubM[i+1]), run_time=1)
                j+=2

            elif i in [3,7] :
                self.play(Write(SubM[i]))

        F = MathTex().add(*tuple(SubM))
        self.add(F)
        self.remove(*tuple(Decomp))
        return F

    def special_case(self, figure1, formula):
        coords_rect=np.array([[np.cos(7*np.pi/6), np.sin(7*np.pi/6), 0],
                             [np.cos(5*np.pi/6), np.sin(5*np.pi/6), 0],
                             [np.cos(np.pi/6), np.sin(np.pi/6), 0],
                             [np.cos(11*np.pi/6), np.sin(11*np.pi/6), 0]])

        rect_angles = [7*np.pi/6, 5*np.pi/6, np.pi/6, 11*np.pi/6]

        rect = Polygon(*tuple(coords_rect), color=vib_green, z_index=-0.5, stroke_width=1.5)
        D1 = Line(coords_rect[0], coords_rect[2], color=YELLOW, z_index=-0.5, stroke_width=1.5)
        D2 = Line(coords_rect[1], coords_rect[3], color=YELLOW, z_index=-0.5, stroke_width=1.5)

        T = []
        for i in range(len(coords_rect)):
            a = Arc(r, angles[i], rect_angles[i]-angles[i], arc_center=figure1[0].get_center())

            T.append(MoveAlongPath(figure1[i+1], a))

        T.append(Transform(figure1[6], D1))
        T.append(Transform(figure1[7], D2))
        T.append(Transform(figure1[5], rect))

        self.play(*tuple(T))

        self.wait(2)

        Pyth = MathTex(r'{{AB^2}} + {{AD^2}} = {{BD^2}}').scale(0.3).move_to(formula.get_center())
        Pyth.submobjects[0].set_color(vib_green)
        Pyth.submobjects[2].set_color(vib_green)
        Pyth.submobjects[4].set_color(YELLOW)

        AB = Line(coords_rect[0], coords_rect[1], color=vib_green, z_index=-0.5, stroke_width=1.5)
        AD = Line(coords_rect[0], coords_rect[3], color=vib_green, z_index=-0.5, stroke_width=1.5)
        BD = Line(coords_rect[1], coords_rect[3], color=YELLOW, z_index=-0.5, stroke_width=1.5)
        R_a = RightAngle(AB, AD, AB.get_arc_length(2)/5, stroke_width=2, quadrant=(1,1))

        self.add(AB,AD,BD)

        self.play(Transform(formula, Pyth))

        self.wait()

        self.play(FadeOut(figure1[6], figure1[3], figure1[5], figure1[0]), FadeIn(R_a))

        self.wait(3)

        self.clean(AB, AD, BD, figure1[7], R_a, formula, figure1[1], figure1[2], figure1[4])
    #956 lines