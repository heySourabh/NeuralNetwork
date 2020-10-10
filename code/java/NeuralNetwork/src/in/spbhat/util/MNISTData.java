/*
 * Copyright (c) 2020. Sourabh Bhat ( https://spbhat.in )
 */

package in.spbhat.util;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.Arrays;

public class MNISTData {
    private final static int W = 28;
    private final static int H = 28;

    private final int label;
    private final double[] data1D;

    public MNISTData(String csvLine) {
        int[] values = Arrays.stream(csvLine.split(","))
                .mapToInt(Integer::parseInt)
                .toArray();

        // first integer is label
        this.label = values[0];

        // remaining integers are gray-scale values in the range [0, 255]
        this.data1D = Arrays.stream(values)
                .skip(1)
                .mapToDouble(v -> v / 255.0)
                .toArray();

        if (data1D.length != (W * H)) {
            throw new IllegalArgumentException("Data size != " + (W * H));
        }
    }

    public void display(double scale) {
        int width = (int) Math.round(W * scale);
        int height = (int) Math.round(H * scale);
        JFrame window = new JFrame("Label = " + label);
        JPanel container = new JPanel() {
            @Override
            public void paint(Graphics g) {
                super.paint(g);
                BufferedImage image = new BufferedImage(W, H, BufferedImage.TYPE_BYTE_GRAY);
                double[][] data2D = getData2D();
                for (int y = 0; y < H; y++) {
                    for (int x = 0; x < W; x++) {
                        final Color c = new Color((float) data2D[y][x], (float) data2D[y][x], (float) data2D[y][x]);
                        image.setRGB(x, y, c.getRGB());
                    }
                }
                Image scaledImage = image.getScaledInstance(width, height, Image.SCALE_REPLICATE);
                g.drawImage(scaledImage, 0, 0, null);
            }
        };
        container.setPreferredSize(new Dimension(width, height));
        window.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
        window.setContentPane(container);
        window.pack();
        window.setVisible(true);
        window.setLocationRelativeTo(null);
    }

    public int getLabel() {
        return label;
    }

    public double[] getData1D() {
        return data1D;
    }

    public double[][] getData2D() {
        double[][] data2D = new double[W][H];

        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                int index = W * i + j;
                data2D[i][j] = data1D[index];
            }
        }
        return data2D;
    }
}
