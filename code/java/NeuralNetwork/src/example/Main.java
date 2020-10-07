package example;

public class Main {
    public static void main(String[] args) {
        Dog snoopie = new Dog("Snoopie", 35);
        snoopie.status();

        snoopie.setTemperature(40);
        snoopie.status();
    }
}
