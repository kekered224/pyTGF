"""
Report and result file writers

Mirrors fprintf logic in TC_GUI.m:
    - L832-L872  : report header (parameters + computation point info)
    - L948-L983  : DEM and density info section
    - L1228-L1250: integration info + output format (itype=10)
    - L1330-L1356: itype=0 output format
    - L1620-L1648: itype=4/103/104 output format
    - save(outname,'tensor','-ASCII') : result matrix output
"""

from typing import List, Optional
import numpy as np


def write_report_header(fid, config, com_data) -> None:
    """
    Write report header: parameters + computation point info.

    Corresponds to TC_GUI.m L832-L872.

    MATLAB:
        fprintf(fid, '=========================    TC_GUI project report   =========================\n');
        fprintf(fid,'\n');
        fprintf(fid,'%30s','Generating_data:'); fprintf(fid,'%14s\n',datestr(now,1));
        fprintf(fid,'%30s','Product_type:');
            if ikind==1  -> 'topographic gravitational field'
            else         -> 'RTM gravitational field'
        fprintf(fid,'%30s','Functional:'); fprintf(fid,'%15s %d %s\n','itype=',itype,'(please refer to handbook)');
        fprintf(fid,'%30s','Gravity field type:'); fprintf(fid,'%15s %d %s\n','ikind=',ikind,'(please refer to handbook)');
        fprintf(fid,'%30s','Earth approximation:'); fprintf(fid,'%15s %d %s\n','iflag=',flag_earth,'(please refer to handbook)');
        fprintf(fid,'%30s','Four zones radius [deg]:'); fprintf(fid,'%7.2f, %7.2f, %7.2f, %7.2f\n',rzones);
        fprintf(fid,'\n');
        fprintf(fid,'----------------------  computation point information  -----------------------\n');
        fprintf(fid,'\n');
        fprintf(fid,'%30s%10.4f\n','comlat_min:',min(com_lat));
        fprintf(fid,'%30s%10.4f\n','comlat_max:',max(com_lat));
        fprintf(fid,'%30s%10.4f\n','comlon_min:',min(com_lon));
        fprintf(fid,'%30s%10.4f\n','comlon_max:',max(com_lon));
        fprintf(fid,'%30s%10d\n','number of computation points:',num);
        fprintf(fid,'\n');

    Args:
        fid: Open file handle (text mode, write).
        config: TGFConfig instance.
        com_data: ComPointData instance.
    """
    w = fid.write

    w('=========================    TC_GUI project report   =========================\n')
    w('\n')
    # MATLAB datestr(now,1) -> 'dd-mmm-yyyy'; use ISO date as a portable equivalent
    import datetime
    w('%30s%14s\n' % ('Generating_data:', datetime.datetime.now().strftime('%d-%b-%Y')))

    w('%30s' % 'Product_type:')
    if config.ikind == 1:
        w('%35s\n' % 'topographic gravitational field')
    else:
        w('%26s\n' % 'RTM gravitational field')

    w('%30s%15s %d %s\n' % ('Functional:', 'itype=', config.itype, '(please refer to handbook)'))
    w('%30s%15s %d %s\n' % ('Gravity field type:', 'ikind=', config.ikind, '(please refer to handbook)'))
    w('%30s%15s %d %s\n' % ('Earth approximation:', 'iflag=', config.flag_earth, '(please refer to handbook)'))
    r1, r2, r3, r4 = config.rzones
    w('%30s%7.2f, %7.2f, %7.2f, %7.2f\n' % ('Four zones radius [deg]:', r1, r2, r3, r4))
    w('\n')

    w('----------------------  computation point information  -----------------------\n')
    w('\n')
    w('%30s%10.4f\n' % ('comlat_min:', float(np.min(com_data.lats))))
    w('%30s%10.4f\n' % ('comlat_max:', float(np.max(com_data.lats))))
    w('%30s%10.4f\n' % ('comlon_min:', float(np.min(com_data.lons))))
    w('%30s%10.4f\n' % ('comlon_max:', float(np.max(com_data.lons))))
    w('%30s%10d\n' % ('number of computation points:', int(com_data.num)))
    w('\n')


def write_report_dem_info(fid, config, dem_layers) -> None:
    """
    Write DEM and density information section.

    Corresponds to TC_GUI.m L948-L983.

    MATLAB:
        fprintf(fid,'-------------------------------  DEM and Mass  -------------------------------\n');
        fprintf(fid,'\n');
        fprintf(fid,'%30s %10.4f\n','DetailDEM_latlimit_north',maxlat1);
        fprintf(fid,'%30s %10.4f\n','DetailDEM_latlimit_south',minlat1);
        fprintf(fid,'%30s %10.4f\n','DetailDEM_reslat',reslat1);
        fprintf(fid,'%30s %10.4f\n','DetailDEM_latlimit_west',minlon1);
        fprintf(fid,'%30s %10.4f\n','DetailDEM_latlimit_est',maxlon1);
        fprintf(fid,'%30s %10.4f\n','DetailDEM_reslon',reslon1);
        fprintf(fid,'\n');
        fprintf(fid,'%30s %10.4f\n','CoarseDEM_latlimit_north',maxlat2);
        ... (CoarseDEM block)
        fprintf(fid,'\n');
        if idensity==0
            fprintf(fid,'%30s %11s %10.4f %10s\n','Density_model:','const_e=',e,'[g/cm^3]');
        else
            fprintf(fid,'%30s %50s','Density_model:','density_model shared the same boundary with DEM');
        fprintf(fid,'\n');

    Args:
        fid: Open file handle.
        config: TGFConfig instance.
        dem_layers: dict with keys 'detailed', 'coarse' (DEMData instances).
    """
    w = fid.write
    detailed = dem_layers['detailed']
    coarse = dem_layers['coarse']

    w('-------------------------------  DEM and Mass  -------------------------------\n')
    w('\n')
    w('%30s %10.4f\n' % ('DetailDEM_latlimit_north', detailed.maxlat))
    w('%30s %10.4f\n' % ('DetailDEM_latlimit_south', detailed.minlat))
    w('%30s %10.4f\n' % ('DetailDEM_reslat', detailed.reslat))
    w('%30s %10.4f\n' % ('DetailDEM_latlimit_west', detailed.minlon))
    w('%30s %10.4f\n' % ('DetailDEM_latlimit_est', detailed.maxlon))
    w('%30s %10.4f\n' % ('DetailDEM_reslon', detailed.reslon))
    w('\n')
    w('%30s %10.4f\n' % ('CoarseDEM_latlimit_north', coarse.maxlat))
    w('%30s %10.4f\n' % ('CoarseDEM_latlimit_south', coarse.minlat))
    w('%30s %10.4f\n' % ('CoarseDEM_reslat', coarse.reslat))
    w('%30s %10.4f\n' % ('CoarseDEM_latlimit_west', coarse.minlon))
    w('%30s %10.4f\n' % ('CoarseDEM_latlimit_est', coarse.maxlon))
    w('%30s %10.4f\n' % ('CoarseDEM_reslon', coarse.reslon))
    w('\n')
    if config.idensity == 0:
        w('%30s %11s %10.4f %10s\n' % ('Density_model:', 'const_e=', config.e, '[g/cm^3]'))
    else:
        w('%30s %50s' % ('Density_model:', 'density_model shared the same boundary with DEM'))
    w('\n')


def write_report_integration(fid, elapsed_hours: float, outname: str,
                              columns: List[str]) -> None:
    """
    Write integration info + output format section.

    Corresponds to TC_GUI.m L1228-L1250 (itype=10), L1330-L1356 (itype=0),
    L1620-L1648 (itype=4/103/104). The structure is identical across itypes;
    only the column header list differs.

    MATLAB (itype=10 example):
        fprintf(fid,'-------------------------  integration information  --------------------------\n');
        fprintf(fid,'\n');
        fprintf(fid,'%30s%10.5f\n','Computation time[hour]:',t);
        fprintf(fid,'\n');
        fprintf(fid,'%30s','Output file name:');
        fprintf(fid,'%15s %s \n','Output file:',outname);
        fprintf(fid,'\n');
        fprintf(fid,'------------------------------  Results Format  ------------------------------\n');
        fprintf(fid,'\n');
        fprintf(fid,'%7s%7s%7s%7s%7s%7s%7s%7s%7s%7s%7s%7s%7s\n', 'lon','lat','H','N','xi','eta','dg','Vxx','Vxy','Vxz','Vyy','Vyz','Vzz');
        fprintf(fid,'%7s%7s%7s%7s%7s%7s%7s%7s%7s%7s%7s%7s%7s\n', '[deg]','[deg]','[m]','[m]','["]','["]','[mGal]','[E]','[E]','[E]','[E]','[E]','[E]');

    Args:
        fid: Open file handle.
        elapsed_hours: Total computation time in hours.
        outname: Output file name.
        columns: List of output column names (e.g. ['lon','lat','H','N']).
    """
    w = fid.write

    w('-------------------------  integration information  --------------------------\n')
    w('\n')
    w('%30s%10.5f\n' % ('Computation time[hour]:', elapsed_hours))
    w('\n')
    w('%30s' % 'Output file name:')
    w('%15s %s \n' % ('Output file:', outname))
    w('\n')
    w('------------------------------  Results Format  ------------------------------\n')
    w('\n')

    # Units row matching each column name (mirrors MATLAB unit strings)
    units_map = {
        'lon': '[deg]', 'lat': '[deg]', 'H': '[m]', 'N': '[m]',
        'xi': '["]', 'eta': '["]', 'dg': '[mGal]',
        'Vxx': '[E]', 'Vxy': '[E]', 'Vxz': '[E]',
        'Vyy': '[E]', 'Vyz': '[E]', 'Vzz': '[E]',
    }
    w(''.join('%7s' % c for c in columns) + '\n')
    w(''.join('%7s' % units_map.get(c, '') for c in columns) + '\n')


def write_result(outname: str, tensor: np.ndarray) -> None:
    """
    Save the result matrix to ASCII file.

    Corresponds to MATLAB: save(outname,'tensor','-ASCII')

    MATLAB save -ASCII writes the matrix with 8 significant digits,
    space-separated, one row per line. numpy savetxt with default fmt
    ('%.18e') differs; to approximate MATLAB's 8-digit output we use
    '%.8e' but keep full precision available. For exact MATLAB parity
    the default '%.18e' is also acceptable since the consumer reads
    generic numeric tables.

    Args:
        outname: Output file path.
        tensor: 2D array of results (n_points x n_columns).
    """
    tensor = np.atleast_2d(np.asarray(tensor, dtype=np.float64))
    np.savetxt(outname, tensor, fmt='%.8e')
