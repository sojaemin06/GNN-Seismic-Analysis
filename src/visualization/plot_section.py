import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

def plot_section_matplotlib(params, output_dir):
    """
    Matplotlib을 사용하여 기둥과 보의 단면 상세도를 그립니다.
    OpenSees 모델 파라미터를 기반으로 정확한 형상을 시각화합니다.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    col_props = params.get('col_props_by_group', {})
    beam_props = params.get('beam_props_by_group', {})
    cover = params.get('cover', 0.04)
    
    # [Helper] Calculate bar diameter from area
    def get_bar_dia(area):
        return np.sqrt(4 * area / np.pi)
    
    # [Helper] Draw grid in a rectangle
    def draw_grid(ax, x_min, y_min, x_max, y_max, n_x, n_y):
        # Vertical lines
        for i in range(1, n_x):
            x = x_min + i * ((x_max - x_min) / n_x)
            ax.plot([x, x], [y_min, y_max], color='black', linewidth=0.5, alpha=0.4)
        # Horizontal lines
        for i in range(1, n_y):
            y = y_min + i * ((y_max - y_min) / n_y)
            ax.plot([x_min, x_max], [y, y], color='black', linewidth=0.5, alpha=0.4)
        # Border (optional, for distinct patches)
        rect = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, 
                                 linewidth=0.5, edgecolor='black', facecolor='none', alpha=0.5)
        ax.add_patch(rect)

    # --- 1. Column Sections ---
    for grp_idx, group_data in col_props.items():
        for loc_type, prop in group_data.items(): # loc_type: 'exterior', 'interior'
            dims = prop['dims'] # (Depth, Width) or (Width, Depth)? -> Z is Width, Y is Depth
            # In model_builder: y_core = dims[1]/2 - cover. So dims[1] is Height(Depth in section).
            # dims[0] is Width.
            # Let's assume dims = (Width_Z, Depth_Y) for visualization consistency with opsvis labels
            # But model_builder uses: 
            # patch rect ... -dims[1]/2 ... 
            # Usually dims in config are [Depth, Width] or [Width, Depth].
            # Config says: "col_section_tiers_m": [[0.5, 0.5], ...]
            # Let's treat dims[0] as Width (Z), dims[1] as Depth (Y).
            
            width = dims[0]
            depth = dims[1]
            rebar = prop['rebar']
            
            nz = rebar['nz'] # Bars along Width (Z)
            nx = rebar['nx'] # Bars along Depth (Y)
            bar_area = rebar['area']
            bar_dia = get_bar_dia(bar_area)
            
            fig, ax = plt.subplots(figsize=(6, 6))
            
            # 1. Concrete Section (Gray Fill with Grid)
            # Outer Concrete
            rect = patches.Rectangle((-width/2, -depth/2), width, depth, 
                                     linewidth=2, edgecolor='black', facecolor='#E0E0E0')
            ax.add_patch(rect)
            
            # [NEW] Grid lines for fibers (Matching model_builder.py 5-patch structure)
            
            # Dimensions
            y_core = depth / 2 - cover
            z_core = width / 2 - cover
            
            # 1. Core Grid (10x10)
            draw_grid(ax, -z_core, -y_core, z_core, y_core, 10, 10)
            
            # 2. Cover Grids
            # Top Cover (Full width, Cover height) - 10x2
            draw_grid(ax, -width/2, y_core, width/2, depth/2, 10, 2)
            # Bottom Cover (Full width, Cover height) - 10x2
            draw_grid(ax, -width/2, -depth/2, width/2, -y_core, 10, 2)
            # Left Cover (Cover width, Core height) - 2x8
            draw_grid(ax, -width/2, -y_core, -z_core, y_core, 2, 8)
            # Right Cover (Cover width, Core height) - 2x8
            draw_grid(ax, z_core, -y_core, width/2, y_core, 2, 8)
            
            # 3. Rebars (Black Circles)
            # Coordinates
            y_core = depth / 2 - cover
            z_core = width / 2 - cover
            
            rebar_coords = []
            
            # Corners
            rebar_coords.append((z_core, y_core))   # Top-Right
            rebar_coords.append((-z_core, y_core))  # Top-Left
            rebar_coords.append((z_core, -y_core))  # Bot-Right
            rebar_coords.append((-z_core, -y_core)) # Bot-Left
            
            # Side Bars (Top/Bot - varying Z)
            if nz > 2:
                s_z = (2 * z_core) / (nz - 1)
                for i in range(1, nz - 1):
                    z_pos = -z_core + i * s_z
                    rebar_coords.append((z_pos, y_core))  # Top
                    rebar_coords.append((z_pos, -y_core)) # Bot
            
            # Side Bars (Left/Right - varying Y)
            if nx > 2:
                s_y = (2 * y_core) / (nx - 1)
                for i in range(1, nx - 1):
                    y_pos = -y_core + i * s_y
                    rebar_coords.append((z_core, y_pos))  # Right
                    rebar_coords.append((-z_core, y_pos)) # Left
            
            # Plot Bars
            for z, y in rebar_coords:
                circle = patches.Circle((z, y), radius=bar_dia/2, color='black', zorder=10)
                ax.add_patch(circle)
            
            # Styling
            ax.set_xlim(-width/2 - 0.1, width/2 + 0.1)
            ax.set_ylim(-depth/2 - 0.1, depth/2 + 0.1)
            ax.set_aspect('equal')
            ax.set_title(f"Col Group {grp_idx} ({loc_type.capitalize()})\n{width:.2f}m(W) x {depth:.2f}m(D)\nMain Bars: {nz}(W) x {nx}(D)")
            ax.set_xlabel("Width (m)")
            ax.set_ylabel("Depth (m)")
            ax.grid(True, linestyle=':', alpha=0.5)
            
            # Save
            filename = f"opsvis_sec_Col_Group{grp_idx}_{loc_type.capitalize()}.png" # Keep old naming convention for compatibility
            plt.savefig(output_dir / filename, dpi=100, bbox_inches='tight')
            plt.close()

    # --- 2. Beam Sections ---
    for grp_idx, group_data in beam_props.items():
        for loc_type, prop in group_data.items():
            dims = prop['dims']
            width = dims[0]
            depth = dims[1]
            rebar = prop['rebar']
            
            num_top = rebar['top']
            num_bot = rebar['bot']
            bar_area = rebar['area']
            bar_dia = get_bar_dia(bar_area)
            
            fig, ax = plt.subplots(figsize=(6, 6))
            
            # Concrete
            rect = patches.Rectangle((-width/2, -depth/2), width, depth, 
                                     linewidth=2, edgecolor='black', facecolor='#F0F0F0')
            ax.add_patch(rect)
            
            # [NEW] Grid lines for fibers (Matching model_builder.py 5-patch structure)
            
            y_core = depth / 2 - cover
            z_core = width / 2 - cover
            
            # 1. Core Grid (10x10)
            draw_grid(ax, -z_core, -y_core, z_core, y_core, 10, 10);
            
            # 2. Cover Grids
            # Top Cover (10x2)
            draw_grid(ax, -width/2, y_core, width/2, depth/2, 10, 2);
            # Bottom Cover (10x2)
            draw_grid(ax, -width/2, -depth/2, width/2, -y_core, 10, 2);
            # Left Cover (2x8)
            draw_grid(ax, -width/2, -y_core, -z_core, y_core, 2, 8);
            # Right Cover (2x8)
            draw_grid(ax, z_core, -y_core, width/2, y_core, 2, 8);
            
            # Top Bars (Distributed along Width)
            if num_top > 1:
                s_top = (2 * z_core) / (num_top - 1)
                for i in range(num_top):
                    z_pos = -z_core + i * s_top
                    circle = patches.Circle((z_pos, y_core), radius=bar_dia/2, color='black', zorder=10)
                    ax.add_patch(circle)
            elif num_top == 1:
                ax.add_patch(patches.Circle((0, y_core), radius=bar_dia/2, color='black', zorder=10))

            # Bottom Bars
            if num_bot > 1:
                s_bot = (2 * z_core) / (num_bot - 1)
                for i in range(num_bot):
                    z_pos = -z_core + i * s_bot
                    circle = patches.Circle((z_pos, -y_core), radius=bar_dia/2, color='black', zorder=10)
                    ax.add_patch(circle)
            elif num_bot == 1:
                ax.add_patch(patches.Circle((0, -y_core), radius=bar_dia/2, color='black', zorder=10))

            ax.set_xlim(-width/2 - 0.1, width/2 + 0.1)
            ax.set_ylim(-depth/2 - 0.1, depth/2 + 0.1)
            ax.set_aspect('equal')
            ax.set_title(f"Beam Group {grp_idx} ({loc_type.capitalize()})\n{width:.2f}m(W) x {depth:.2f}m(D)\nTop: {num_top}, Bot: {num_bot}")
            ax.set_xlabel("Width (m)")
            ax.set_ylabel("Depth (m)")
            ax.grid(True, linestyle=':', alpha=0.5)
            
            filename = f"opsvis_sec_Beam_Group{grp_idx}_{loc_type.capitalize()}.png"
            plt.savefig(output_dir / filename, dpi=100, bbox_inches='tight')
            plt.close()

    print(f"Section plots saved to {output_dir}")
