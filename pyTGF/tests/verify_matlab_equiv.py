"""
MATLAB-Python 等效性验证脚本

用一组固定输入跑 Python 实现，打印所有输出。
同时生成等效的 MATLAB 验证脚本 verify_matlab.m，用户可在 MATLAB 中
跑相同输入来对比数值结果。

运行: python -m pyTGF.tests.verify_matlab_equiv
"""

import numpy as np
from pyTGF.config import PhysicalConstants
from pyTGF.mass_elements.point_mass import point_mass
from pyTGF.mass_elements.tesseroid import tesseroid
from pyTGF.mass_elements.prism import prism


def run_python_verification():
    """用固定输入跑 Python 实现，打印所有输出。"""
    a = PhysicalConstants.A
    G = PhysicalConstants.G
    Pi = PhysicalConstants.PI
    d2r = Pi / 180.0

    print("=" * 70)
    print("Python 实现验证 (point_mass + tesseroid + prism)")
    print("=" * 70)
    print(f"Constants: G={G}, Pi={Pi}, d2r={d2r}, A={a}")
    print()

    # ----- Test case 1: single tesseroid/point_mass -----
    print("--- Test case 1: single element ---")
    lon_c, lat_c = 116.5, 40.5
    ref = a - 2000.0       # lower radius
    h1 = a - 1000.0        # upper radius
    e = np.array([2.67])
    lon, lat, r0 = 116.0, 40.0, a + 500.0

    # point_mass (bounds format)
    deg = 0.01
    xx1 = np.array([lon_c - deg / 2])
    xx2 = np.array([lon_c + deg / 2])
    yy1 = np.array([lat_c - deg / 2])
    yy2 = np.array([lat_c + deg / 2])
    r1 = np.array([h1])
    r2 = np.array([ref])

    g, Vx, Vy, Vz, Vxx, Vxy, Vxz, Vyy, Vyz, Vzz = point_mass(
        xx1, xx2, yy1, yy2, r1, r2, lon, lat, r0, e)
    print(f"point_mass inputs:")
    print(f"  xx1={xx1[0]}, xx2={xx2[0]}, yy1={yy1[0]}, yy2={yy2[0]}")
    print(f"  r1={r1[0]}, r2={r2[0]}, lon={lon}, lat={lat}, r0={r0}, e={e[0]}")
    print(f"point_mass outputs:")
    print(f"  g  = {g:.15e}")
    print(f"  Vx = {Vx:.15e}")
    print(f"  Vy = {Vy:.15e}")
    print(f"  Vz = {Vz:.15e}")
    print(f"  Vxx= {Vxx:.15e}")
    print(f"  Vxy= {Vxy:.15e}")
    print(f"  Vxz= {Vxz:.15e}")
    print(f"  Vyy= {Vyy:.15e}")
    print(f"  Vyz= {Vyz:.15e}")
    print(f"  Vzz= {Vzz:.15e}")
    print()

    # tesseroid (center format)
    V, Vx, Vy, Vz, Vxx, Vxy, Vxz, Vyy, Vyz, Vzz = tesseroid(
        np.array([lon_c]), np.array([lat_c]), np.array([h1]), np.array([ref]),
        lon, lat, r0, deg, deg, e)
    print(f"tesseroid inputs:")
    print(f"  lon1={lon_c}, lat1={lat_c}, h1={h1}, ref={ref}")
    print(f"  lon={lon}, lat={lat}, r0={r0}, rx={deg}, ry={deg}, e={e[0]}")
    print(f"tesseroid outputs:")
    print(f"  V  = {V:.15e}")
    print(f"  Vx = {Vx:.15e}")
    print(f"  Vy = {Vy:.15e}")
    print(f"  Vz = {Vz:.15e}")
    print(f"  Vxx= {Vxx:.15e}")
    print(f"  Vxy= {Vxy:.15e}")
    print(f"  Vxz= {Vxz:.15e}")
    print(f"  Vyy= {Vyy:.15e}")
    print(f"  Vyz= {Vyz:.15e}")
    print(f"  Vzz= {Vzz:.15e}")
    print()

    # prism (bounds format, same as point_mass)
    # d1 = h1 (upper radius, top of mass), d2 = ref (lower radius, bottom)
    g, Vx, Vy, Vz, Vxx, Vxy, Vxz, Vyy, Vyz, Vzz = prism(
        xx1, xx2, yy1, yy2, r1, r2, lon, lat, r0, e, deg, deg)
    print(f"prism inputs:")
    print(f"  x1={xx1[0]}, x2={xx2[0]}, y1={yy1[0]}, y2={yy2[0]}")
    print(f"  d1={r1[0]}, d2={r2[0]}, lon={lon}, lat={lat}, r0={r0}, e={e[0]}")
    print(f"  resolution_x={deg}, resolution_y={deg}")
    print(f"prism outputs:")
    print(f"  g  = {g:.15e}")
    print(f"  Vx = {Vx:.15e}")
    print(f"  Vy = {Vy:.15e}")
    print(f"  Vz = {Vz:.15e}")
    print(f"  Vxx= {Vxx:.15e}")
    print(f"  Vxy= {Vxy:.15e}")
    print(f"  Vxz= {Vxz:.15e}")
    print(f"  Vyy= {Vyy:.15e}")
    print(f"  Vyz= {Vyz:.15e}")
    print(f"  Vzz= {Vzz:.15e}")
    print()

    # ----- Test case 2: array of 3 elements -----
    print("--- Test case 2: array of 3 elements ---")
    lon_c = np.array([116.5, 117.0, 115.5])
    lat_c = np.array([40.5, 41.0, 39.5])
    ref = np.array([a - 2000.0, a - 1500.0, a - 2500.0])
    h1 = np.array([a - 1000.0, a - 500.0, a - 1200.0])
    e = np.array([2.67, 2.70, 2.65])
    lon, lat, r0 = 116.0, 40.0, a + 500.0
    deg = 0.02

    # point_mass
    xx1 = lon_c - deg / 2
    xx2 = lon_c + deg / 2
    yy1 = lat_c - deg / 2
    yy2 = lat_c + deg / 2
    r1 = h1
    r2 = ref

    g, Vx, Vy, Vz, Vxx, Vxy, Vxz, Vyy, Vyz, Vzz = point_mass(
        xx1, xx2, yy1, yy2, r1, r2, lon, lat, r0, e)
    print(f"point_mass (3 elements) outputs:")
    print(f"  g  = {g:.15e}")
    print(f"  Vx = {Vx:.15e}")
    print(f"  Vy = {Vy:.15e}")
    print(f"  Vz = {Vz:.15e}")
    print(f"  Vxx= {Vxx:.15e}")
    print(f"  Vxy= {Vxy:.15e}")
    print(f"  Vxz= {Vxz:.15e}")
    print(f"  Vyy= {Vyy:.15e}")
    print(f"  Vyz= {Vyz:.15e}")
    print(f"  Vzz= {Vzz:.15e}")
    print()

    # tesseroid
    V, Vx, Vy, Vz, Vxx, Vxy, Vxz, Vyy, Vyz, Vzz = tesseroid(
        lon_c, lat_c, h1, ref, lon, lat, r0, deg, deg, e)
    print(f"tesseroid (3 elements) outputs:")
    print(f"  V  = {V:.15e}")
    print(f"  Vx = {Vx:.15e}")
    print(f"  Vy = {Vy:.15e}")
    print(f"  Vz = {Vz:.15e}")
    print(f"  Vxx= {Vxx:.15e}")
    print(f"  Vxy= {Vxy:.15e}")
    print(f"  Vxz= {Vxz:.15e}")
    print(f"  Vyy= {Vyy:.15e}")
    print(f"  Vyz= {Vyz:.15e}")
    print(f"  Vzz= {Vzz:.15e}")
    print()

    # prism (3 elements)
    g, Vx, Vy, Vz, Vxx, Vxy, Vxz, Vyy, Vyz, Vzz = prism(
        xx1, xx2, yy1, yy2, r1, r2, lon, lat, r0, e, deg, deg)
    print(f"prism (3 elements) outputs:")
    print(f"  g  = {g:.15e}")
    print(f"  Vx = {Vx:.15e}")
    print(f"  Vy = {Vy:.15e}")
    print(f"  Vz = {Vz:.15e}")
    print(f"  Vxx= {Vxx:.15e}")
    print(f"  Vxy= {Vxy:.15e}")
    print(f"  Vxz= {Vxz:.15e}")
    print(f"  Vyy= {Vyy:.15e}")
    print(f"  Vyz= {Vyz:.15e}")
    print(f"  Vzz= {Vzz:.15e}")
    print()


def generate_matlab_script():
    """生成等效的 MATLAB 验证脚本。"""
    a = PhysicalConstants.A
    script = f"""%% MATLAB 验证脚本 - 与 Python verify_matlab_equiv.py 对比
% 运行后对比输出值，应与 Python 输出完全一致（到机器精度）
%
% 注意: Point_mass / Tesseroid / Prism / global2local 等是 TC_GUI.m 的子函数，
%       无法直接调用。本脚本将它们内联，独立运行无需 addpath。

G = 6.67384e-3;
Pi = 3.141592654;
d2r = Pi/180;
A = {a};

disp('=== Test case 1: single element ===');

% --- point_mass ---
lon_c = 116.5; lat_c = 40.5;
ref = A - 2000.0;
h1 = A - 1000.0;
e = 2.67;
lon = 116.0; lat = 40.0; r0 = A + 500.0;
deg = 0.01;

xx1 = lon_c - deg/2;
xx2 = lon_c + deg/2;
yy1 = lat_c - deg/2;
yy2 = lat_c + deg/2;
r1 = h1;
r2 = ref;

[g, Vx, Vy, Vz, Vxx, Vxy, Vxz, Vyy, Vyz, Vzz] = ...
    local_Point_mass(xx1, xx2, yy1, yy2, r1, r2, lon, lat, r0, e);
fprintf('point_mass:\\n');
fprintf('  g  = %.15e\\n', g);
fprintf('  Vx = %.15e\\n', Vx);
fprintf('  Vy = %.15e\\n', Vy);
fprintf('  Vz = %.15e\\n', Vz);
fprintf('  Vxx= %.15e\\n', Vxx);
fprintf('  Vxy= %.15e\\n', Vxy);
fprintf('  Vxz= %.15e\\n', Vxz);
fprintf('  Vyy= %.15e\\n', Vyy);
fprintf('  Vyz= %.15e\\n', Vyz);
fprintf('  Vzz= %.15e\\n', Vzz);

% --- tesseroid ---
lon1 = lon_c; lat1 = lat_c;
[V, Vx, Vy, Vz, Vxx, Vxy, Vxz, Vyy, Vyz, Vzz] = ...
    local_Tesseroid(lon1, lat1, h1, ref, lon, lat, r0, deg, deg, e);
fprintf('tesseroid:\\n');
fprintf('  V  = %.15e\\n', V);
fprintf('  Vx = %.15e\\n', Vx);
fprintf('  Vy = %.15e\\n', Vy);
fprintf('  Vz = %.15e\\n', Vz);
fprintf('  Vxx= %.15e\\n', Vxx);
fprintf('  Vxy= %.15e\\n', Vxy);
fprintf('  Vxz= %.15e\\n', Vxz);
fprintf('  Vyy= %.15e\\n', Vyy);
fprintf('  Vyz= %.15e\\n', Vyz);
fprintf('  Vzz= %.15e\\n', Vzz);

% --- prism ---
% prism uses bounds format: x1,x2 (lon), y1,y2 (lat), d1=h1 (upper), d2=ref (lower)
[g, Vx, Vy, Vz, Vxx, Vxy, Vxz, Vyy, Vyz, Vzz] = ...
    local_Prism(xx1, xx2, yy1, yy2, r1, r2, lon, lat, r0, e, deg, deg);
fprintf('prism:\\n');
fprintf('  g  = %.15e\\n', g);
fprintf('  Vx = %.15e\\n', Vx);
fprintf('  Vy = %.15e\\n', Vy);
fprintf('  Vz = %.15e\\n', Vz);
fprintf('  Vxx= %.15e\\n', Vxx);
fprintf('  Vxy= %.15e\\n', Vxy);
fprintf('  Vxz= %.15e\\n', Vxz);
fprintf('  Vyy= %.15e\\n', Vyy);
fprintf('  Vyz= %.15e\\n', Vyz);
fprintf('  Vzz= %.15e\\n', Vzz);

disp(' ');
disp('=== Test case 2: array of 3 elements ===');

% --- point_mass (3 elements) ---
lon_c = [116.5, 117.0, 115.5];
lat_c = [40.5, 41.0, 39.5];
ref = [A-2000.0, A-1500.0, A-2500.0];
h1 = [A-1000.0, A-500.0, A-1200.0];
e = [2.67, 2.70, 2.65];
lon = 116.0; lat = 40.0; r0 = A + 500.0;
deg = 0.02;

xx1 = lon_c - deg/2;
xx2 = lon_c + deg/2;
yy1 = lat_c - deg/2;
yy2 = lat_c + deg/2;
r1 = h1;
r2 = ref;

[g, Vx, Vy, Vz, Vxx, Vxy, Vxz, Vyy, Vyz, Vzz] = ...
    local_Point_mass(xx1, xx2, yy1, yy2, r1, r2, lon, lat, r0, e);
fprintf('point_mass (3 elements):\\n');
fprintf('  g  = %.15e\\n', g);
fprintf('  Vx = %.15e\\n', Vx);
fprintf('  Vy = %.15e\\n', Vy);
fprintf('  Vz = %.15e\\n', Vz);
fprintf('  Vxx= %.15e\\n', Vxx);
fprintf('  Vxy= %.15e\\n', Vxy);
fprintf('  Vxz= %.15e\\n', Vxz);
fprintf('  Vyy= %.15e\\n', Vyy);
fprintf('  Vyz= %.15e\\n', Vyz);
fprintf('  Vzz= %.15e\\n', Vzz);

% --- tesseroid (3 elements) ---
lon1 = lon_c; lat1 = lat_c;
[V, Vx, Vy, Vz, Vxx, Vxy, Vxz, Vyy, Vyz, Vzz] = ...
    local_Tesseroid(lon1, lat1, h1, ref, lon, lat, r0, deg, deg, e);
fprintf('tesseroid (3 elements):\\n');
fprintf('  V  = %.15e\\n', V);
fprintf('  Vx = %.15e\\n', Vx);
fprintf('  Vy = %.15e\\n', Vy);
fprintf('  Vz = %.15e\\n', Vz);
fprintf('  Vxx= %.15e\\n', Vxx);
fprintf('  Vxy= %.15e\\n', Vxy);
fprintf('  Vxz= %.15e\\n', Vxz);
fprintf('  Vyy= %.15e\\n', Vyy);
fprintf('  Vyz= %.15e\\n', Vyz);
fprintf('  Vzz= %.15e\\n', Vzz);

% --- prism (3 elements) ---
[g, Vx, Vy, Vz, Vxx, Vxy, Vxz, Vyy, Vyz, Vzz] = ...
    local_Prism(xx1, xx2, yy1, yy2, r1, r2, lon, lat, r0, e, deg, deg);
fprintf('prism (3 elements):\\n');
fprintf('  g  = %.15e\\n', g);
fprintf('  Vx = %.15e\\n', Vx);
fprintf('  Vy = %.15e\\n', Vy);
fprintf('  Vz = %.15e\\n', Vz);
fprintf('  Vxx= %.15e\\n', Vxx);
fprintf('  Vxy= %.15e\\n', Vxy);
fprintf('  Vxz= %.15e\\n', Vxz);
fprintf('  Vyy= %.15e\\n', Vyy);
fprintf('  Vyz= %.15e\\n', Vyz);
fprintf('  Vzz= %.15e\\n', Vzz);


%% ====== 以下为内联函数 (从 TC_GUI.m 复制) ======

function [x, y, z]=global2local(lon,lat,r,prism_lon,prism_lat,prism_r)
% transform the spherical coordinates of station to prism-based local cartesian coordinates
Pi=3.141592654;
d2r=Pi/180;

prism_lon=d2r*prism_lon;
prism_lat=d2r*prism_lat;
lon=d2r*lon;
lat=d2r*lat;

coslat=cos(lat);
coslon=cos(lon);
sinlat=sin(lat);
sinlon=sin(lon);

cosprism_lon=cos(prism_lon);
cosprism_lat=cos(prism_lat);
sinprism_lon=sin(prism_lon);
sinprism_lat=sin(prism_lat);

g_x = r.*coslat.*coslon -prism_r.*cosprism_lat.*cosprism_lon;
g_y= r.*coslat.*sinlon - prism_r.*cosprism_lat.*sinprism_lon;
g_z= r.*sinlat - prism_r.*sinprism_lat;

x = -g_x.*sinprism_lat.*cosprism_lon - g_y.*sinprism_lat.*sinprism_lon + g_z.*cosprism_lat;
y = -g_x.*sinprism_lon +  g_y.*cosprism_lon;
z = -(g_x.*cosprism_lat.*cosprism_lon + g_y.*cosprism_lat.*sinprism_lon + g_z.*sinprism_lat);
end


function  [g, Vx, Vy, Vz, Vxx, Vxy, Vxz, Vyy, Vyz, Vzz] = local_Point_mass(xx1,xx2,yy1,yy2,r1,r2,lon,lat,r0,e)
% calculation of all gravitational functionals generated by point-mass
G = 6.67384e-3;   % Universal Gravitational constant
d2r=3.141592654/180;

point_r=(r1+r2)/2;
point_lon=(xx1+xx2)/2;
point_lat=(yy1+yy2)/2;

% tesseroid equivalent masses
mass=d2r*e.*(xx2-xx1).*(r1.^3-r2.^3).*(sin(d2r*yy2)-sin(d2r*yy1))/3;

% spherical coordinates to local cartesian coordinates
[dx dy dz]=global2local(point_lon,point_lat,point_r,lon,lat,r0);

l = 1./(sqrt(dx.*dx+dy.*dy+dz.*dz));
l3 = l.*l.*l;
l5 = l3.*l.*l;
c=G*mass;

res = c.*l;
g=-sum(res)*1e-5;

res=c.*(dx.*l3);
gx=res;

res=c.*(dy.*l3);
gy=res;

res=c.*(dz.*l3);
gz=-res;

Vx=sum(gx);
Vy=sum(gy);
Vz=sum(gz);

res = c.*(-l3 + 3*dx.*dx.*l5);
gxx=-res;
res = c.*(3*dx.*dy.*l5);
gxy=-res;
res = c.*(3*dx.*dz.*l5);
gxz=res;
res = c.*(-l3 + 3*dy.*dy.*l5);
gyy=-res;
res = c.*(3*dy.*dz.*l5);
gyz=res;
res = c.*(-l3 + 3*dz.*dz.*l5);
gzz=-res;

Vxx=sum(gxx)*1e4;
Vxy=sum(gxy)*1e4;
Vxz=sum(gxz)*1e4;
Vyy=sum(gyy)*1e4;
Vyz=sum(gyz)*1e4;
Vzz=sum(gzz)*1e4;
end


function [V, Vx, Vy, Vz, Vxx, Vxy, Vxz, Vyy, Vyz, Vzz]=local_Tesseroid(lon1,lat1,h1,ref,lon,lat,r0,resolution_x1,resolution_y1,e)
Pi=3.141592654;
d2r=Pi/180;
G = 6.67384e-3;   % Universal Gravitational constant

%% build tesseroid
zz1=lon1-resolution_x1/2;
zz2=lon1+resolution_x1/2;
yy1=lat1-resolution_y1/2;
yy2=lat1+resolution_y1/2;
xx1=h1;
xx2=ref;

r1=(xx1+xx2)/2;
%% variable definition
lat=lat*d2r;
lat1=lat1*d2r;
coslat=cos(lat);
coslat1=cos(lat1);
sinlat=sin(lat);
sinlat1=sin(lat1);
d_lon=(lon1-lon)*d2r;
cosdlon=cos(d_lon);
sindlon=sin(d_lon);
cosphicosa=coslat1.*cosdlon;
coslatsinlat1=coslat.*sinlat1;
sinphicosa=coslatsinlat1-sinlat.*cosphicosa;
sinphisina=coslat1.*sindlon;
sinlatsinlat1=sinlat.*sinlat1;
cosphi=sinlatsinlat1+coslat.*cosphicosa;
dx=r1.*sinphicosa;
dy=r1.*sinphisina;
dz=r1.*cosphi-r0;
dxx=dx.*dx;
dxy=dx.*dy;
dxz=dx.*dz;
dyy=dy.*dy;
dyz=dy.*dz;
dzz=dz.*dz;
l=sqrt(dxx+dyy+dzz);
xx=xx2-xx1;
yy=(yy2-yy1)*d2r;
zz=(zz2-zz1)*d2r;
c=G*e.*zz.*xx.*yy;
l2=l.*l;
%% Toi
r12=r1.*r1;
toi=r12.*coslat1;
toi1=2*r1.*coslat1;
toi2=-r12.*sinlat1;
toi3=0;
toi12=2*coslat1;
toi22=-toi;
toi32=0;
%% dxi
x1_1=sinphicosa;
x2_1=sinphisina;
x3_1=cosphi;
x1_12=0;
x2_12=0;
x3_12=0;
x1_2=r1.*(coslat.*coslat1+sinlatsinlat1.*cosdlon);
x2_2=-r1.*sinlat1.*sindlon;
x3_2=r1.*(sinlat.*coslat1-coslatsinlat1.*cosdlon);
x1_22=-dx;
x2_22=-dy;
x3_22=-r1.*cosphi;
x1_3=(dy).*sinlat;
x2_3=r1.*cosphicosa;
x3_3=-(dy).*coslat;
x1_32=x2_3.*sinlat;
x2_32=-(dy);
x3_32=-x2_3.*coslat;
%% ml
ml1=r0.*cosphi-r1;
ml2=r0.*x3_2;
ml3=r0.*x3_3;
ml12=-1;
ml22=r0.*x3_22;
ml32=r0.*x3_32;
%% a
a=l2;
a1=dx;
a2=dy;
a3=dz;
l23=3./l2;
a11=l23.*dxx-1;
a12=l23.*dxy;
a13=l23.*dxz;
a22=l23.*dyy-1;
a23=l23.*dyz;
a33=l23.*dzz-1;
%% b
b1=2*ml1;
b2=2*ml2;
b3=2*ml3;
b11=2*(x1_1+l23.*dx.*ml1);
b12=2*(x1_2+l23.*dx.*ml2);
b21=2*(x2_1+l23.*dy.*ml1);
b13=2*(x1_3+l23.*dx.*ml3);
b31=2*(x3_1+l23.*dz.*ml1);
b22=2*(x2_2+l23.*dy.*ml2);
b23=2*(x2_3+l23.*dy.*ml3);
b32=2*(x3_2+l23.*dz.*ml2);
b33=2*(x3_3+l23.*dz.*ml3);
const_1=6./l2;
dxx5=5*dxx;
dxy5=5*dxy;
dxz5=5*dxz;
dyy5=5*dyy;
dyz5=5*dyz;
dzz5=5*dzz;
dx_x11=dx.*x1_1;
dx_x12=dx.*x1_2;
dx_x13=dx.*x1_3;
dx_x21=dx.*x2_1;
dx_x22=dx.*x2_2;
dx_x23=dx.*x2_3;
dy_x11=dy.*x1_1;
dy_x12=dy.*x1_2;
dy_x13=dy.*x1_3;
dz_x11=dz.*x1_1;
dz_x12=dz.*x1_2;
dz_x13=dz.*x1_3;
dy_x21=dy.*x2_1;
dy_x22=dy.*x2_2;
dy_x23=dy.*x2_3;
dz_x21=dz.*x2_1;
dz_x22=dz.*x2_2;
dz_x23=dz.*x2_3;
dx_x31=dx.*x3_1;
dx_x32=dx.*x3_2;
dx_x33=dx.*x3_3;
dy_x31=dy.*x3_1;
dy_x32=dy.*x3_2;
dy_x33=dy.*x3_3;
dz_x31=dz.*x3_1;
dz_x32=dz.*x3_2;
dz_x33=dz.*x3_3;
b111=const_1.*(dx_x11+dx_x11+dxx5.*ml1./l2-ml1);
b112=const_1.*(dx_x12+dx_x12+dxx5.*ml2./l2-ml2);
b113=const_1.*(dx_x13+dx_x13+dxx5.*ml3./l2-ml3);
b121=const_1.*(dx_x21+dy_x11+dxy5.*ml1./l2);
b122=const_1.*(dx_x22+dy_x12+dxy5.*ml2./l2);
b123=const_1.*(dx_x23+dy_x13+dxy5.*ml3./l2);
b131=const_1.*(dx_x31+dz_x11+dxz5.*ml1./l2);
b132=const_1.*(dx_x32+dz_x12+dxz5.*ml2./l2);
b133=const_1.*(dx_x33+dz_x13+dxz5.*ml3./l2);
b221=const_1.*(dy_x21+dy_x21+dyy5.*ml1./l2-ml1);
b222=const_1.*(dy_x22+dy_x22+dyy5.*ml2./l2-ml2);
b223=const_1.*(dy_x23+dy_x23+dyy5.*ml3./l2-ml3);
b231=const_1.*(dy_x31+dz_x21+dyz5.*ml1./l2);
b232=const_1.*(dy_x32+dz_x22+dyz5.*ml2./l2);
b233=const_1.*(dy_x33+dz_x23+dyz5.*ml3./l2);
b331=const_1.*(dz_x31+dz_x31+dzz5.*ml1./l2-ml1);
b332=const_1.*(dz_x32+dz_x32+dzz5.*ml2./l2-ml2);
b333=const_1.*(dz_x33+dz_x33+dzz5.*ml3./l2-ml3);
%% gama
ml1_2=ml1.*ml1;
ml2_2=ml2.*ml2;
ml3_2=ml3.*ml3;
gama1=ml12+l23.*ml1_2;
gama2=ml22+l23.*ml2_2;
gama3=ml32+l23.*ml3_2;
l25=5./l2;
gama11=x1_12+l23.*(2*x1_1.*ml1+dx.*(ml12+l25.*ml1_2));
gama12=x1_22+l23.*(2*x1_2.*ml2+dx.*(ml22+l25.*ml2_2));
gama21=x2_12+l23.*(2*x2_1.*ml1+dy.*(ml12+l25.*ml1_2));
gama13=x1_32+l23.*(2*x1_3.*ml3+dx.*(ml32+l25.*ml3_2));
gama31=x3_12+l23.*(2*x3_1.*ml1+dz.*(ml12+l25.*ml1_2));
gama22=x2_22+l23.*(2*x2_2.*ml2+dy.*(ml22+l25.*ml2_2));
gama23=x2_32+l23.*(2*x2_3.*ml3+dy.*(ml32+l25.*ml3_2));
gama32=x3_22+l23.*(2*x3_2.*ml2+dz.*(ml22+l25.*ml2_2));
gama33=x3_32+l23.*(2*x3_3.*ml3+dz.*(ml32+l25.*ml3_2));
l27=7./l2;
dx_x112=dx.*x1_12;
dx_x122=dx.*x1_22;
dx_x132=dx.*x1_32;
dx_x212=dx.*x2_12;
dx_x222=dx.*x2_22;
dx_x232=dx.*x2_32;
dx_x312=dx.*x3_12;
dx_x322=dx.*x3_22;
dx_x332=dx.*x3_32;
dy_x112=dy.*x1_12;
dy_x122=dy.*x1_22;
dy_x132=dy.*x1_32;
dy_x212=dy.*x2_12;
dy_x222=dy.*x2_22;
dy_x232=dy.*x2_32;
dy_x312=dy.*x3_12;
dy_x322=dy.*x3_22;
dy_x332=dy.*x3_32;
dz_x112=dz.*x1_12;
dz_x122=dz.*x1_22;
dz_x132=dz.*x1_32;
dz_x212=dz.*x2_12;
dz_x222=dz.*x2_22;
dz_x232=dz.*x2_32;
dz_x312=dz.*x3_12;
dz_x322=dz.*x3_22;
dz_x332=dz.*x3_32;
gama111=l23.*(dx_x112+dx_x112+2*x1_1.*x1_1+l25.*(b1.*(dx_x11+dx_x11)+dxx.*(ml12+l27.*(ml1_2)))-(ml12+l25.*(ml1_2)));
gama112=l23.*(dx_x122+dx_x122+2*x1_2.*x1_2+l25.*(b2.*(dx_x12+dx_x12)+dxx.*(ml22+l27.*(ml2_2)))-(ml22+l25.*(ml2_2)));
gama113=l23.*(dx_x132+dx_x132+2*x1_3.*x1_3+l25.*(b3.*(dx_x13+dx_x13)+dxx.*(ml32+l27.*(ml3_2)))-(ml32+l25.*(ml3_2)));
gama121=l23.*(dx_x212+dy_x112+2*x1_1.*x2_1+l25.*(b1.*(dx_x21+dy_x11)+dxy.*(ml12+l27.*(ml1_2))));
gama122=l23.*(dx_x222+dy_x122+2*x1_2.*x2_2+l25.*(b2.*(dx_x22+dy_x12)+dxy.*(ml22+l27.*(ml2_2))));
gama123=l23.*(dx_x232+dy_x132+2*x1_3.*x2_3+l25.*(b3.*(dx_x23+dy_x13)+dxy.*(ml32+l27.*(ml3_2))));
gama131=l23.*(dx_x312+dz_x112+2*x1_1.*x3_1+l25.*(b1.*(dx_x31+dz_x11)+dxz.*(ml12+l27.*(ml1_2))));
gama132=l23.*(dx_x322+dz_x122+2*x1_2.*x3_2+l25.*(b2.*(dx_x32+dz_x12)+dxz.*(ml22+l27.*(ml2_2))));
gama133=l23.*(dx_x332+dz_x132+2*x1_3.*x3_3+l25.*(b3.*(dx_x33+dz_x13)+dxz.*(ml32+l27.*(ml3_2))));
gama221=l23.*(dy_x212+dy_x212+2*x2_1.*x2_1+l25.*(b1.*(dy_x21+dy_x21)+dyy.*(ml12+l27.*(ml1_2)))-(ml12+l25.*(ml1_2)));
gama222=l23.*(dy_x222+dy_x222+2*x2_2.*x2_2+l25.*(b2.*(dy_x22+dy_x22)+dyy.*(ml22+l27.*(ml2_2)))-(ml22+l25.*(ml2_2)));
gama223=l23.*(dy_x232+dy_x232+2*x2_3.*x2_3+l25.*(b3.*(dy_x23+dy_x23)+dyy.*(ml32+l27.*(ml3_2)))-(ml32+l25.*(ml3_2)));
gama231=l23.*(dy_x312+dz_x212+2*x2_1.*x3_1+l25.*(b1.*(dy_x31+dz_x21)+dyz.*(ml12+l27.*(ml1_2))));
gama232=l23.*(dy_x322+dz_x222+2*x2_2.*x3_2+l25.*(b2.*(dy_x32+dz_x22)+dyz.*(ml22+l27.*(ml2_2))));
gama233=l23.*(dy_x332+dz_x232+2*x2_3.*x3_3+l25.*(b3.*(dy_x33+dz_x23)+dyz.*(ml32+l27.*(ml3_2))));
gama331=l23.*(dz_x312+dz_x312+2*x3_1.*x3_1+l25.*(b1.*(dz_x31+dz_x31)+dzz.*(ml12+l27.*(ml1_2)))-(ml12+l25.*(ml1_2)));
gama332=l23.*(dz_x322+dz_x322+2*x3_2.*x3_2+l25.*(b2.*(dz_x32+dz_x32)+dzz.*(ml22+l27.*(ml2_2)))-(ml22+l25.*(ml2_2)));
gama333=l23.*(dz_x332+dz_x332+2*x3_3.*x3_3+l25.*(b3.*(dz_x33+dz_x33)+dzz.*(ml32+l27.*(ml3_2)))-(ml32+l25.*(ml3_2)));
%% gravitational field
xx_2=(xx).*xx;
yy_2=(yy).*yy;
zz_2=(zz).*zz;
K000=1./l.*toi;
l3=1./l.^3;
g=c.*(K000+1/24.*l3.*(xx_2.*(toi12.*a+toi1.*b1+toi.*gama1)+yy_2.*(toi22.*a+toi2.*b2+toi.*gama2)+zz_2.*(toi32.*a+toi3.*b3+toi.*gama3)));
V=sum(g(:))*1e-5;

Lx=l3.*dx.*toi;
Ly=l3.*dy.*toi;
Lz=l3.*dz.*toi;

gx=c.*(Lx+1/24.*l3.*(xx_2.*(toi12.*a1+toi1.*b11+toi.*gama11)+yy_2.*(toi22.*a1+toi2.*b12+toi.*gama12)+zz_2.*(toi32.*a1+toi3.*b13+toi.*gama13)));
gy=c.*(Ly+1/24.*l3.*(xx_2.*(toi12.*a2+toi1.*b21+toi.*gama21)+yy_2.*(toi22.*a2+toi2.*b22+toi.*gama22)+zz_2.*(toi32.*a2+toi3.*b23+toi.*gama23)));
gz=c.*(Lz+1/24.*l3.*(xx_2.*(toi12.*a3+toi1.*b31+toi.*gama31)+yy_2.*(toi22.*a3+toi2.*b32+toi.*gama32)+zz_2.*(toi32.*a3+toi3.*b33+toi.*gama33)));
Vx=-sum(gx(:));
Vy=-sum(gy(:));
Vz=-sum(gz(:));

Kxx=l3.*(l23.*dxx-1).*toi;
Kxy=l3.*(l23.*dxy).*toi;
Kxz=l3.*(l23.*dxz).*toi;
Kyy=l3.*(l23.*dyy-1).*toi;
Kyz=l3.*(l23.*dyz).*toi;
Kzz=l3.*(l23.*dzz-1).*toi;
Lxx=c.*(Kxx+1/24.*l3.*(xx_2.*(toi12.*a11+toi1.*b111+toi.*gama111)+yy_2.*(toi22.*a11+toi2.*b112+toi.*gama112)+zz_2.*(toi32.*a11+toi3.*b113+toi.*gama113)));
Vxx=sum(Lxx(:))*1e4;
Lxy=c.*(Kxy+1/24.*l3.*(xx_2.*(toi12.*a12+toi1.*b121+toi.*gama121)+yy_2.*(toi22.*a12+toi2.*b122+toi.*gama122)+zz_2.*(toi32.*a12+toi3.*b123+toi.*gama123)));
Vxy=sum(Lxy(:))*1e4;
Lxz=c.*(Kxz+1/24.*l3.*(xx_2.*(toi12.*a13+toi1.*b131+toi.*gama131)+yy_2.*(toi22.*a13+toi2.*b132+toi.*gama132)+zz_2.*(toi32.*a13+toi3.*b133+toi.*gama133)));
Vxz=sum(Lxz(:))*1e4;
Lyy=c.*(Kyy+1/24.*l3.*(xx_2.*(toi12.*a22+toi1.*b221+toi.*gama221)+yy_2.*(toi22.*a22+toi2.*b222+toi.*gama222)+zz_2.*(toi32.*a22+toi3.*b223+toi.*gama223)));
Vyy=sum(Lyy(:))*1e4;
Lyz=c.*(Kyz+1/24.*l3.*(xx_2.*(toi12.*a23+toi1.*b231+toi.*gama231)+yy_2.*(toi22.*a23+toi2.*b232+toi.*gama232)+zz_2.*(toi32.*a23+toi3.*b233+toi.*gama233)));
Vyz=sum(Lyz(:))*1e4;
Lzz=c.*(Kzz+1/24.*l3.*(xx_2.*(toi12.*a33+toi1.*b331+toi.*gama331)+yy_2.*(toi32.*a33+toi2.*b332+toi.*gama332)+zz_2.*(toi32.*a33+toi3.*b333+toi.*gama333)));
Vzz=sum(Lzz(:))*1e4;
end


function [g, Vx, Vy, Vz, Vxx, Vxy, Vxz, Vyy, Vyz, Vzz]=local_Prism(x1,x2,y1,y2,d1,d2,lon,lat,r0,e,resolution_x,resolution_y)
% calculation of all gravitational functionals generated by prism
% (inlined from TC_GUI.m Prism, L3277-L3567)
G = 6.67384e-3;
Pi=3.141592654;
d2r=Pi/180;

% definitation of prism and prism-based Cartesian coordinate system
[prism_x1 prism_x2 prism_y1 prism_y2 prism_z1 prism_z2 prism_lon prism_lat prism_r prism_e]=local_tess2prism(x1,x2,y1,y2,d1,d2,e);

c = G*prism_e;  % multiplication of gravitational constant and density

[x y z]=global2local(lon,lat,r0,prism_lon,prism_lat,prism_r);

xpos2=(prism_x2-x);
xpos1=(prism_x1-x);
ypos2=(prism_y2-y);
ypos1=(prism_y1-y);
zpos2=(prism_z1-z);
zpos1=(prism_z2-z);

% detecting irregular points
fx1=abs(xpos1)<=0.05;
nx1=find(fx1);
fx2=abs(xpos2)<=0.05;
nx2=find(fx2);
fy2=abs(ypos2)<=0.05;
ny2=find(fy2);
fy1=abs(ypos1)<=0.05;
ny1=find(fy1);
fz2=abs(zpos2)<=0.05;
nz2=find(fz2);
fz1=abs(zpos1)<=0.05;
nz1=find(fz1);

r1 = sqrt(xpos1.*xpos1 + ypos1.*ypos1 + (zpos1).*(zpos1));
r2 = sqrt(xpos2.*xpos2 + ypos1.*ypos1 + (zpos1).*(zpos1));
r3 = sqrt(xpos1.*xpos1 + ypos2.*ypos2 + (zpos1).*(zpos1));
r4 = sqrt(xpos2.*xpos2 + ypos2.*ypos2 + (zpos1).*(zpos1));
r5 = sqrt(xpos1.*xpos1 + ypos1.*ypos1 + (zpos2).*(zpos2));
r6 = sqrt(xpos2.*xpos2 + ypos1.*ypos1 + (zpos2).*(zpos2));
r7 = sqrt(xpos1.*xpos1 + ypos2.*ypos2 + (zpos2).*(zpos2));
r8 = sqrt(xpos2.*xpos2 + ypos2.*ypos2 + (zpos2).*(zpos2));

%% g
res = 0;
[t1 t2 t3 t4 t5 t6]=local_g6term(r1,xpos1,ypos1,zpos1,nx1,ny1,nz1);
value =(t1+t2+t3+t4+t5+t6);
res=res+value;
[t1 t2 t3 t4 t5 t6]=local_g6term(r2,xpos2,ypos1,zpos1,nx2,ny1,nz1);
value =(t1+t2+t3+t4+t5+t6);
res=res-value;
[t1 t2 t3 t4 t5 t6]=local_g6term(r3,xpos1,ypos2,zpos1,nx1,ny2,nz1);
value =(t1+t2+t3+t4+t5+t6);
res=res-value;
[t1 t2 t3 t4 t5 t6]=local_g6term(r4,xpos2,ypos2,zpos1,nx2,ny2,nz1);
value =(t1+t2+t3+t4+t5+t6);
res=res+value;
[t1 t2 t3 t4 t5 t6]=local_g6term(r5,xpos1,ypos1,zpos2,nx1,ny1,nz2);
value =(t1+t2+t3+t4+t5+t6);
res=res-value;
[t1 t2 t3 t4 t5 t6]=local_g6term(r6,xpos2,ypos1,zpos2,nx2,ny1,nz2);
value =(t1+t2+t3+t4+t5+t6);
res=res+value;
[t1 t2 t3 t4 t5 t6]=local_g6term(r7,xpos1,ypos2,zpos2,nx1,ny2,nz2);
value =(t1+t2+t3+t4+t5+t6);
res=res+value;
[t1 t2 t3 t4 t5 t6]=local_g6term(r8,xpos2,ypos2,zpos2,nx2,ny2,nz2);
value =(t1+t2+t3+t4+t5+t6);
res=res-value;
g= -sum(c.*res)*1e-5;

%% gz
s1=xpos1.*log((ypos1+r1).*(ypos2+r7)./((ypos2+r3).*(ypos1+r5)));
s1(nx1)=0;
s2=xpos2.*log((ypos1+r2).*(ypos2+r8)./((ypos2+r4).*(ypos1+r6)));
s2(nx2)=0;
s=s1-s2;
s1=ypos1.*log((xpos1+r1).*(xpos2+r6)./((xpos2+r2).*(xpos1+r5)));
s1(ny1)=0;
s2=ypos2.*log((xpos1+r3).*(xpos2+r8)./((xpos2+r4).*(xpos1+r7)));
s2(ny2)=0;
value=s+s1-s2;
[t3]=local_gz3term(r1,xpos1,ypos1,zpos1,nz1);
value=value+t3;
[t3]=local_gz3term(r2,xpos2,ypos1,zpos1,nz1);
value=value-t3;
[t3]=local_gz3term(r3,xpos1,ypos2,zpos1,nz1);
value=value-t3;
[t3]=local_gz3term(r4,xpos2,ypos2,zpos1,nz1);
value=value+t3;
[t3]=local_gz3term(r5,xpos1,ypos1,zpos2,nz2);
value=value-t3;
[t3]=local_gz3term(r6,xpos2,ypos1,zpos2,nz2);
value=value+t3;
[t3]=local_gz3term(r7,xpos1,ypos2,zpos2,nz2);
value=value+t3;
[t3]=local_gz3term(r8,xpos2,ypos2,zpos2,nz2);
value=value-t3;
gz=c.*value;

%% gy
s1=xpos1.*log((zpos1+r1).*(zpos2+r7)./((zpos1+r3).*(zpos2+r5)));
s1(nx1)=0;
s2=xpos2.*log((zpos1+r2).*(zpos2+r8)./((zpos1+r4).*(zpos2+r6)));
s2(nx2)=0;
s=s1-s2;
s1=zpos1.*log((xpos1+r1).*(xpos2+r4)./((xpos2+r2).*(xpos1+r3)));
s1(nz1)=0;
s2=zpos2.*log((xpos1+r5).*(xpos2+r8)./((xpos2+r6).*(xpos1+r7)));
s2(nz2)=0;
value=s+s1-s2;
[t3]=local_gz3term(r1,xpos1,zpos1,ypos1,ny1);
value=value+t3;
[t3]=local_gz3term(r2,xpos2,zpos1,ypos1,ny1);
value=value-t3;
[t3]=local_gz3term(r3,xpos1,zpos1,ypos2,ny2);
value=value-t3;
[t3]=local_gz3term(r4,xpos2,zpos1,ypos2,ny2);
value=value+t3;
[t3]=local_gz3term(r5,xpos1,zpos2,ypos1,ny1);
value=value-t3;
[t3]=local_gz3term(r6,xpos2,zpos2,ypos1,ny1);
value=value+t3;
[t3]=local_gz3term(r7,xpos1,zpos2,ypos2,ny2);
value=value+t3;
[t3]=local_gz3term(r8,xpos2,zpos2,ypos2,ny2);
value=value-t3;
gy=-c.*value;

%% gx
s1=ypos1.*log((zpos1+r1).*(zpos2+r6)./((zpos1+r2).*(zpos2+r5)));
s1(ny1)=0;
s2=ypos2.*log((zpos1+r3).*(zpos2+r8)./((zpos1+r4).*(zpos2+r7)));
s2(ny2)=0;
s=s1-s2;
s1=zpos1.*log((ypos1+r1).*(ypos2+r4)./((ypos1+r2).*(ypos2+r3)));
s1(nz1)=0;
s2=zpos2.*log((ypos1+r5).*(ypos2+r8)./((ypos2+r7).*(ypos1+r6)));
s2(nz2)=0;
value=s+s1-s2;
[t3]=local_gz3term(r1,ypos1,zpos1,xpos1,nx1);
value=value+t3;
[t3]=local_gz3term(r2,ypos1,zpos1,xpos2,nx2);
value=value-t3;
[t3]=local_gz3term(r3,ypos2,zpos1,xpos1,nx1);
value=value-t3;
[t3]=local_gz3term(r4,ypos2,zpos1,xpos2,nx2);
value=value+t3;
[t3]=local_gz3term(r5,ypos1,zpos2,xpos1,nx1);
value=value-t3;
[t3]=local_gz3term(r6,ypos1,zpos2,xpos2,nx2);
value=value+t3;
[t3]=local_gz3term(r7,ypos2,zpos2,xpos1,nx1);
value=value+t3;
[t3]=local_gz3term(r8,ypos2,zpos2,xpos2,nx2);
value=value-t3;
gx=-c.*value;

%% Vxx
res = 0;
valuexx =atan((zpos1).*(ypos1)./((xpos1).*r1));
valuexx(nx1)=0;
res=res+valuexx;
valuexx =atan((zpos1).*(ypos1)./((xpos2).*r2));
valuexx(nx2)=0;
res=res-valuexx;
valuexx =atan((zpos1).*(ypos2)./((xpos1).*r3));
valuexx(nx1)=0;
res=res-valuexx;
valuexx =atan((zpos1).*(ypos2)./((xpos2).*r4));
valuexx(nx2)=0;
res=res+valuexx;
valuexx =atan((zpos2).*(ypos1)./((xpos1).*r5));
valuexx(nx1)=0;
res=res-valuexx;
valuexx =atan((zpos2).*(ypos1)./((xpos2).*r6));
valuexx(nx2)=0;
res=res+valuexx;
valuexx =atan((zpos2).*(ypos2)./((xpos1).*r7));
valuexx(nx1)=0;
res=res+valuexx;
valuexx =atan((zpos2).*(ypos2)./((xpos2).*r8));
valuexx(nx2)=0;
res=res-valuexx;
gxx = (c.*res)*1e4;

%% Vyy
res = 0;
valuexx =atan((zpos1).*(xpos1)./((ypos1).*r1));
valuexx(ny1)=0;
res=res+valuexx;
valuexx =atan((zpos1).*(xpos2)./((ypos1).*r2));
valuexx(ny1)=0;
res=res-valuexx;
valuexx =atan((zpos1).*(xpos1)./((ypos2).*r3));
valuexx(ny2)=0;
res=res-valuexx;
valuexx =atan((zpos1).*(xpos2)./((ypos2).*r4));
valuexx(ny2)=0;
res=res+valuexx;
valuexx =atan((zpos2).*(xpos1)./((ypos1).*r5));
valuexx(ny1)=0;
res=res-valuexx;
valuexx =atan((zpos2).*(xpos2)./((ypos1).*r6));
valuexx(ny1)=0;
res=res+valuexx;
valuexx =atan((zpos2).*(xpos1)./((ypos2).*r7));
valuexx(ny2)=0;
res=res+valuexx;
valuexx =atan((zpos2).*(xpos2)./((ypos2).*r8));
valuexx(ny2)=0;
res=res-valuexx;
gyy = (c.*res)*1e4;

%% Vzz
res = 0;
valuezz =atan((xpos1).*(ypos1)./((zpos1).*r1));
valuezz(nz1)=0;
res=res+valuezz;
valuezz =atan((xpos2).*(ypos1)./((zpos1).*r2));
valuezz(nz1)=0;
res=res-valuezz;
valuezz =atan((xpos1).*(ypos2)./((zpos1).*r3));
valuezz(nz1)=0;
res=res-valuezz;
valuezz =atan((xpos2).*(ypos2)./((zpos1).*r4));
valuezz(nz1)=0;
res=res+valuezz;
valuezz =atan((xpos1).*(ypos1)./((zpos2).*r5));
valuezz(nz2)=0;
res=res-valuezz;
valuezz =atan((xpos2).*(ypos1)./((zpos2).*r6));
valuezz(nz2)=0;
res=res+valuezz;
valuezz =atan((xpos1).*(ypos2)./((zpos2).*r7));
valuezz(nz2)=0;
res=res+valuezz;
valuezz =atan((xpos2).*(ypos2)./((zpos2).*r8));
valuezz(nz2)=0;
res=res-valuezz;
gzz = (c.*res)*1e4;

%% Vxy
s1=(r1+zpos1).*(r4+zpos1).*(r6+zpos2).*(r7+zpos2);
s2=(r2+zpos1).*(r3+zpos1).*(r5+zpos2).*(r8+zpos2);
res=log(s1./s2);
n=find(abs(s2)<=0.001 | abs(s1)<=0.001);
res(n)=0;
gxy = -c.*(res)*1e4;

%% Vxz
s1=(r1+ypos1).*(r4+ypos2).*(r6+ypos1).*(r7+ypos2);
s2=(r2+ypos1).*(r3+ypos2).*(r5+ypos1).*(r8+ypos2);
res=log(s1./s2);
n=find(abs(s2)<=0.001 | abs(s1)<=0.001);
res(n)=0;
gxz = c.*(res)*1e4;

%% Vyz
s1=(r1+xpos1).*(r4+xpos2).*(r6+xpos2).*(r7+xpos1);
s2=(r2+xpos2).*(r3+xpos1).*(r5+xpos1).*(r8+xpos2);
res=log(s1./s2);
n=find(abs(s2)<=0.001 | abs(s1)<=0.001);
res(n)=0;
gyz = c.*(res)*1e4;

% gravitaional field transformation from prism-based system to station-based Cartesian coordinate system
[temp_Vx temp_Vy temp_Vz]=local_g_prism2point(gx,gy,gz,lon,lat,prism_lon,prism_lat);
Vx=sum(temp_Vx(:));
Vy=sum(temp_Vy(:));
Vz=sum(temp_Vz(:));

[temp_Vxx temp_Vxy temp_Vxz temp_Vyy temp_Vyz temp_Vzz]=local_ggt_prism2point(gxx,gxy,gxz,gyy,gyz,gzz,lon,lat,prism_lon,prism_lat);
Vxx=sum(temp_Vxx(:));
Vxy=sum(temp_Vxy(:));
Vxz=sum(temp_Vxz(:));
Vyy=sum(temp_Vyy(:));
Vyz=sum(temp_Vyz(:));
Vzz=sum(temp_Vzz(:));
end


function [prism_x1, prism_x2, prism_y1, prism_y2, prism_z1, prism_z2, prism_lon, prism_lat, prism_r, prism_e]=local_tess2prism(x1,x2,y1,y2,d1,d2,e)
% transform a tesseroid to a mass-equavilent prism
Pi=3.141592654;
d2r=Pi/180;

r1=d1;
r2=d2;
r0 = 0.5*(r1 + r2);

dx=r0*d2r.*(y2-y1);
dy=r0.*cos(d2r*0.5*(y1+y2))*d2r.*(x2-x1);
prism_x1=-0.5*dx;
prism_x2=-prism_x1;
prism_y1=-0.5*dy;
prism_y2=-prism_y1;
prism_z1=r1;
prism_z1(1:end)=(r2-r1)/2;
prism_z2=(r1-r2)/2;

tess_volume=d2r.*(x2-x1).*(r1.^3-r2.^3).*(sin(d2r*y2)-sin(d2r*y1))/3;
prism_volume=(prism_x2-prism_x1).*(prism_y2-prism_y1).*(prism_z2-prism_z1);

prism_e=e.*tess_volume./prism_volume;
n=find(prism_volume==0);
prism_e(n)=e(n);
prism_lon=0.5*(x1+x2);
prism_lat=0.5*(y1+y2);
prism_r=r0;
end


function [Vx, Vy, Vz]=local_g_prism2point(gx,gy,gz,lon,lat,prism_lon,prism_lat)
Pi=3.141592654;
d2r = Pi/180;

prism_lon=d2r*prism_lon;
prism_lat=d2r*prism_lat;
lon=d2r*lon;
lat=d2r*lat;

cosbeta = cos((prism_lon - lon));
sinbeta = sin((prism_lon - lon));
cosphi = cos(lat);
sinphi = sin(lat);
cosphil = cos(prism_lat);
sinphil = sin(prism_lat);

cosbetasinphi= cosbeta.*sinphi;
cosbetacosphi= cosbeta.*cosphi;
R0 = cosbetasinphi.*sinphil + cosphi.*cosphil;
R1 = sinbeta.*sinphi;
R2 = -cosbetasinphi.*cosphil + cosphi.*sinphil;
R3 = -sinbeta.*sinphil;
R4= cosbeta;
R5 = sinbeta.*cosphil;
R6 = -cosbetacosphi.*sinphil + sinphi.*cosphil;
R7 = -sinbeta.*cosphi;
R8 = cosbetacosphi.*cosphil + sinphi.*sinphil;

Vx=R0.*gx+R1.*gy+R2.*gz;
Vy=R3.*gx+R4.*gy+R5.*gz;
Vz=R6.*gx+R7.*gy+R8.*gz;
end


function [Vxx, Vxy, Vxz, Vyy, Vyz, Vzz]=local_ggt_prism2point(gxx,gxy,gxz,gyy,gyz,gzz,lon,lat,prism_lon,prism_lat)
Pi=3.141592654;
d2r = Pi/180;

atprism0=gxx;
atprism1=gxy;
atprism2=gxz;
atprism3=gxy;
atprism4=gyy;
atprism5=gyz;
atprism6=gxz;
atprism7=gyz;
atprism8=gzz;

prism_lon=d2r*prism_lon;
prism_lat=d2r*prism_lat;
lon=d2r*lon;
lat=d2r*lat;

cosbeta = cos((prism_lon - lon));
sinbeta = sin((prism_lon - lon));
cosphi = cos(lat);
sinphi = sin(lat);
cosphil = cos(prism_lat);
sinphil = sin(prism_lat);

cosbetasinphi=cosbeta.*sinphi;
cosbetacosphi=cosbeta.*cosphi;
R0 = cosbetasinphi.*sinphil + cosphi.*cosphil;
R1 = sinbeta.*sinphi;
R2 = -cosbetasinphi.*cosphil + cosphi.*sinphil;
R3 = -sinbeta.*sinphil;
R4= cosbeta;
R5 = sinbeta.*cosphil;
R6 = -cosbetacosphi.*sinphil + sinphi.*cosphil;
R7 = -sinbeta.*cosphi;
R8 = cosbetacosphi.*cosphil + sinphi.*sinphil;

t0=R0.*atprism0+R1.*atprism3+R2.*atprism6;
t1=R0.*atprism1+R1.*atprism4+R2.*atprism7;
t2=R0.*atprism2+R1.*atprism5+R2.*atprism8;
t3=R3.*atprism0+R4.*atprism3+R5.*atprism6;
t4=R3.*atprism1+R4.*atprism4+R5.*atprism7;
t5=R3.*atprism2+R4.*atprism5+R5.*atprism8;
t6=R6.*atprism0+R7.*atprism3+R8.*atprism6;
t7=R6.*atprism1+R7.*atprism4+R8.*atprism7;
t8=R6.*atprism2+R7.*atprism5+R8.*atprism8;

V0=t0.*R0+t1.*R1+t2.*R2;
V1=t0.*R3+t1.*R4+t2.*R5;
V2=t0.*R6+t1.*R7+t2.*R8;
V4=t3.*R3+t4.*R4+t5.*R5;
V7=t6.*R3+t7.*R4+t8.*R5;
V8=t6.*R6+t7.*R7+t8.*R8;

Vxx=V0;
Vxy=V1;
Vxz=V2;
Vyy=V4;
Vyz=V7;
Vzz=V8;
end


function [t3]=local_gz3term(r,xpos1,ypos1,zpos1,nz1)
t3=-(zpos1).*atan(ypos1.*xpos1./((zpos1).*r));
t3(nz1)=0;
end


function [t1, t2,t3, t4, t5, t6]=local_g6term(r,xpos1,ypos1,zpos1,nx1,ny1,nz1)
t1=xpos1.*ypos1.*log(zpos1+r);
t1(nx1)=0;
t1(ny1)=0;

t2=ypos1.*(zpos1).*log(xpos1+r);
t2(ny1)=0;
t2(nz1)=0;

t3=xpos1.*(zpos1).*log(ypos1+r);
t3(nx1)=0;
t3(nz1)=0;

t4=-xpos1.^2/2.*atan(ypos1.*(zpos1)./(xpos1.*r));
t4(nx1)=0;

t5=-ypos1.^2/2.*atan(xpos1.*(zpos1)./(ypos1.*r));
t5(ny1)=0;

t6=-(zpos1).^2/2.*atan(ypos1.*xpos1./((zpos1).*r));
t6(nz1)=0;
end
"""
    path = r"d:\Trae_re\TGF_matlab\pyTGF\tests\verify_matlab.m"
    with open(path, "w", encoding="utf-8") as f:
        f.write(script)
    print(f"MATLAB 验证脚本已生成: {path}")
    print("在 MATLAB 中运行此脚本，对比输出值。")


if __name__ == "__main__":
    run_python_verification()
    generate_matlab_script()
