package example;

public class Dog {
    private final String name;
    private double temperature;

    public Dog(String name, double temperature) {
        this.name = name;
        this.temperature = temperature;
    }

    public void setTemperature(double temperature) {
        this.temperature = temperature;
    }

    public void status() {
        System.out.println(toString());
    }

    public void bark() {
        System.out.println("woof! woof!");
    }

    @Override
    public String toString() {
        return "The dog's name is '" + name + "'\n" +
                "The dog's temperature is " + temperature + " deg. Celsius.";
    }
}
