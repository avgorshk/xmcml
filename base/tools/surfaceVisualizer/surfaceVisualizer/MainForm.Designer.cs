namespace surfaceVisualizer
{
    partial class MainForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.MainMenu = new System.Windows.Forms.MenuStrip();
            this.fileToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.opentxtFileToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripMenuItem1 = new System.Windows.Forms.ToolStripSeparator();
            this.closeToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.OGL = new Tao.Platform.Windows.SimpleOpenGlControl();
            this.opensurfaceFileToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.MainMenu.SuspendLayout();
            this.SuspendLayout();
            // 
            // MainMenu
            // 
            this.MainMenu.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.fileToolStripMenuItem});
            this.MainMenu.Location = new System.Drawing.Point(0, 0);
            this.MainMenu.Name = "MainMenu";
            this.MainMenu.Size = new System.Drawing.Size(463, 24);
            this.MainMenu.TabIndex = 0;
            this.MainMenu.Text = "menuStrip1";
            // 
            // fileToolStripMenuItem
            // 
            this.fileToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.opentxtFileToolStripMenuItem,
            this.opensurfaceFileToolStripMenuItem,
            this.toolStripMenuItem1,
            this.closeToolStripMenuItem});
            this.fileToolStripMenuItem.Name = "fileToolStripMenuItem";
            this.fileToolStripMenuItem.Size = new System.Drawing.Size(37, 20);
            this.fileToolStripMenuItem.Text = "File";
            // 
            // opentxtFileToolStripMenuItem
            // 
            this.opentxtFileToolStripMenuItem.Name = "opentxtFileToolStripMenuItem";
            this.opentxtFileToolStripMenuItem.Size = new System.Drawing.Size(171, 22);
            this.opentxtFileToolStripMenuItem.Text = "Open *.txt file";
            this.opentxtFileToolStripMenuItem.Click += new System.EventHandler(this.opentxtFileToolStripMenuItem_Click);
            // 
            // toolStripMenuItem1
            // 
            this.toolStripMenuItem1.Name = "toolStripMenuItem1";
            this.toolStripMenuItem1.Size = new System.Drawing.Size(168, 6);
            // 
            // closeToolStripMenuItem
            // 
            this.closeToolStripMenuItem.Name = "closeToolStripMenuItem";
            this.closeToolStripMenuItem.Size = new System.Drawing.Size(171, 22);
            this.closeToolStripMenuItem.Text = "Close";
            this.closeToolStripMenuItem.Click += new System.EventHandler(this.closeToolStripMenuItem_Click);
            // 
            // OGL
            // 
            this.OGL.AccumBits = ((byte)(0));
            this.OGL.AutoCheckErrors = false;
            this.OGL.AutoFinish = false;
            this.OGL.AutoMakeCurrent = true;
            this.OGL.AutoSwapBuffers = true;
            this.OGL.BackColor = System.Drawing.Color.Black;
            this.OGL.ColorBits = ((byte)(32));
            this.OGL.DepthBits = ((byte)(16));
            this.OGL.Dock = System.Windows.Forms.DockStyle.Fill;
            this.OGL.Location = new System.Drawing.Point(0, 24);
            this.OGL.Name = "OGL";
            this.OGL.Size = new System.Drawing.Size(463, 304);
            this.OGL.StencilBits = ((byte)(0));
            this.OGL.TabIndex = 1;
            this.OGL.Paint += new System.Windows.Forms.PaintEventHandler(this.OGL_Paint);
            this.OGL.KeyPress += new System.Windows.Forms.KeyPressEventHandler(this.OGL_KeyPress);
            this.OGL.Resize += new System.EventHandler(this.OGL_Resize);
            // 
            // opensurfaceFileToolStripMenuItem
            // 
            this.opensurfaceFileToolStripMenuItem.Name = "opensurfaceFileToolStripMenuItem";
            this.opensurfaceFileToolStripMenuItem.Size = new System.Drawing.Size(171, 22);
            this.opensurfaceFileToolStripMenuItem.Text = "Open *.surface file";
            this.opensurfaceFileToolStripMenuItem.Click += new System.EventHandler(this.opensurfaceFileToolStripMenuItem_Click);
            // 
            // MainForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(463, 328);
            this.Controls.Add(this.OGL);
            this.Controls.Add(this.MainMenu);
            this.MainMenuStrip = this.MainMenu;
            this.Name = "MainForm";
            this.Text = "Surface Visualizer";
            this.Load += new System.EventHandler(this.MainForm_Load);
            this.MainMenu.ResumeLayout(false);
            this.MainMenu.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.MenuStrip MainMenu;
        private System.Windows.Forms.ToolStripMenuItem fileToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem closeToolStripMenuItem;
        private Tao.Platform.Windows.SimpleOpenGlControl OGL;
        private System.Windows.Forms.ToolStripMenuItem opentxtFileToolStripMenuItem;
        private System.Windows.Forms.ToolStripSeparator toolStripMenuItem1;
        private System.Windows.Forms.ToolStripMenuItem opensurfaceFileToolStripMenuItem;
    }
}

